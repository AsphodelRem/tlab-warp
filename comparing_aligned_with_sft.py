import math
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from tqdm import tqdm

from dataset_utils import get_datasets_for_warp, get_non_overlapping_subsamples_for_warp
from config_utils import load_config
from warp import WarpTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_aligned_model', type=str)
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    # Load SFT and aligned models
    tokenizer = AutoTokenizer.from_pretrained(
        args.path_to_aligned_model,
        use_fast=True, 
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Get WarpTrainer class just for using generate_completion and compute_reward functions
    warp_trainer = WarpTrainer(config, None)

    # Get aligned and sft models
    aligned_model = AutoModelForCausalLM.from_pretrained(
        args.path_to_aligned_model
    ).to(warp_trainer.device)
    
    sft_model = AutoModelForCausalLM.from_pretrained(
        config['warp']['sft_model_name_or_path']
    ).to(warp_trainer.device)

    # Get dataset
    test_sample = get_non_overlapping_subsamples_for_warp(config)

    aligned_model_mean_rewards = []
    sft_model_mean_rewards = []
    kl_divs = []
    for sample in tqdm(test_sample):
        # Get generations
        with torch.no_grad():
            aligned_model_output, aligned_logprobs = warp_trainer._generate_completion(
                aligned_model, 
                warp_trainer.sft_tokenizer, 
                sample
            )
            
            sft_model_output, sft_logprobs = warp_trainer._generate_completion(
                sft_model, 
                warp_trainer.sft_tokenizer, 
                sample
            )
    
        # Get rewards
        with torch.no_grad():
            aligned_model_reward = warp_trainer._compute_reward(
                warp_trainer.reward_model, 
                warp_trainer.reward_model_tokenizer, 
                aligned_model_output
            )

        with torch.no_grad():
            sft_model_reward = warp_trainer._compute_reward(
                warp_trainer.reward_model, 
                warp_trainer.reward_model_tokenizer, 
                sft_model_output
            )

        aligned_model_mean_rewards.append(torch.mean(aligned_model_reward))
        sft_model_mean_rewards.append(torch.mean(sft_model_reward))
        kl_divs.append(torch.nn.functional.kl_div(aligned_logprobs, sft_logprobs, log_target=True, reduction='batchmean'))

    aligned_model_mean_rewards = torch.tensor(aligned_model_mean_rewards)
    sft_model_mean_rewards = torch.tensor(sft_model_mean_rewards)
    print(f'Aligned model mean reward: {torch.mean(aligned_model_mean_rewards)}')
    print(f'SFT model mean reward: {torch.mean(sft_model_mean_rewards)}')

    print(f'RMSE: {torch.sqrt(torch.mean((aligned_model_mean_rewards - sft_model_mean_rewards)**2))}')

    print(f'Mean KL: {torch.mean(torch.tensor(kl_divs))}')