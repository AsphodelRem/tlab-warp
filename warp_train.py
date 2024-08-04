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

    # Get dataset
    train_dataset, test_dataset = get_datasets_for_warp(config)

    # Train
    warp_trainer = WarpTrainer(
        config,
        train_dataset
    )

    warp_trainer.train()
    warp_trainer.save_model()
