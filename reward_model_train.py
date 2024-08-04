import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import (
    RewardTrainer,
    RewardConfig,
)

from dataset_utils import get_datasets_for_reward_model
from config_utils import load_config


"""
Usage:
    python3 reward_model_train.py --config path/to/config.toml
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    
    reward_config = RewardConfig(
        output_dir=config['reward_model_trainer']['output_dir'],
        per_device_train_batch_size=config['reward_model_trainer']['per_device_train_batch_size'],
        num_train_epochs=config['reward_model_trainer']['num_train_epochs'],
        gradient_checkpointing=config['reward_model_trainer']['gradient_checkpointing'],
        remove_unused_columns=config['reward_model_trainer']['remove_unused_columns'],
        max_length=config['reward_model_trainer']['max_length']
    )

    # Create a model
    tokenizer = AutoTokenizer.from_pretrained(
        config['reward_model_trainer']['model_name_or_path'], use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config['reward_model_trainer']['model_name_or_path'], num_labels=1
    )

    # Get train/test split of the dataset
    train_dataset, test_dataset = get_datasets_for_reward_model(config, tokenizer)

    # Train
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
