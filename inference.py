import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from dataset_utils import get_datasets_for_warp
from config_utils import load_config


def generate_text(model, tokenizer, prompt):
    model_input = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**model_input, max_length=20)
    decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    print(decoded_outputs)
    

"""
Usage:
    python3 inference.py \
        --config path/to/config.toml \
        --user_input True
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_aligned_model', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--user_input', type=str)
    args = parser.parse_args()

    config = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(
        config['warp']['output_dir'],
        use_fast=True, 
        padding_side='left',
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config['warp']['output_dir']
    )

    _, test_dataset = get_datasets_for_warp(config)

    if args.user_input:
        while True:
            user_prompt = str(input())
            generate_text(model, tokenizer, user_prompt)
            
    else: 
        generate_text(model, tokenizer, test_dataset)