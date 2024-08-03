import torch
import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM
)

from warp import WarpTrainer
from dataset_utils import get_datasets_for_warp
from config_utils import load_config


"""
Usage:
    python3 warp_train.py --config path/to/config.toml
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    
    gpt_tokenizer = AutoTokenizer.from_pretrained(
        config['warp']['sft_model_name_or_path'],
        use_fast=True, 
        padding_side='left',
    )
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    
    gpt_model = AutoModelForCausalLM.from_pretrained(
       config['warp']['sft_model_name_or_path']
    )
    reward_model_tokenizer = AutoTokenizer.from_pretrained(
        config['warp']['reward_model'], use_fast=True
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config['warp']['reward_model'], num_labels=1
    )

    # Get dataset
    train_dataset, test_dataset = get_datasets_for_warp()

    # Train
    warp_trainer = WarpTrainer(
        gpt_model,
        gpt_tokenizer,
        reward_model, 
        reward_model_tokenizer,
        config,
        train_dataset
    )

    warp_trainer.train()
    warp_trainer.save_model()