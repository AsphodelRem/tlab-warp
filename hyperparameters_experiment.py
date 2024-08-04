import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)

from warp import WarpTrainer
from dataset_utils import get_datasets_for_warp, get_non_overlapping_subsamples_for_warp
from config_utils import load_config


"""
Studying the impact of number of training steps on mean reward and KL. 
Chosen values = 50, 100, 150.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    
    config = load_config(args.config)    

    train_dataset, _ = get_datasets_for_warp(config)
    test_samples = get_non_overlapping_subsamples_for_warp(config)

    training_steps = [50, 100, 150]
    for steps in training_steps:
        print(f'Number of steps: {steps}')
        config['warp']['training_steps'] = steps
        warp_trainer = WarpTrainer(config, train_dataset)

        sft_model = AutoModelForCausalLM.from_pretrained(
            config['warp']['sft_model_name_or_path']
        ).to(warp_trainer.device)
        
        warp_trainer.train()

        # Measuring values on 5 random subsamples
        aligned_model_mean_rewards, kl_divs = [], []
        for sample in tqdm(test_samples):
            with torch.no_grad():
                aligned_model_output, aligned_logprobs = warp_trainer._generate_completion(
                    warp_trainer.sft, 
                    warp_trainer.sft_tokenizer, 
                    sample
                )
            with torch.no_grad():
                sft_model_output, sft_logprobs = warp_trainer._generate_completion(
                    sft_model, 
                    warp_trainer.sft_tokenizer, 
                    sample
                )
                
            with torch.no_grad():
                aligned_model_reward = warp_trainer._compute_reward(
                    warp_trainer.reward_model, 
                    warp_trainer.reward_model_tokenizer, 
                    aligned_model_output
                )
                
            aligned_model_mean_rewards.append(torch.mean(aligned_model_reward))
            kl_divs.append(
                torch.nn.functional.kl_div(
                    aligned_logprobs, 
                    sft_logprobs, 
                    log_target=True, 
                    reduction='batchmean'
                )
            )

        print(f'Aligned model mean reward: {torch.mean(torch.tensor(aligned_model_mean_rewards))}')
        print(f'Mean KL: {torch.mean(torch.tensor(kl_divs))}')
        torch.cuda.empty_cache()