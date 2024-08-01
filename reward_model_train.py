from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    RewardTrainer,
    RewardConfig,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from dataset_utils import get_datasets

"""
Usage:
python3 reward_model_train.py \
    --model_name_or_path="distilbert/distilbert-base-cased" \
    --output_dir="d_bert" \
    --per_device_train_batch_size=64 \
    --per_device_test_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_checkpointing=True \
    --remove_unused_columns=False \
    --max_length=512
"""

if __name__ == "__main__":
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, bert_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Create a model
    tokenizer = AutoTokenizer.from_pretrained(
        bert_config.model_name_or_path, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_config.model_name_or_path, num_labels=1
    )

    # Get train/test split of the dataset
    train_dataset, test_dataset = get_datasets(tokenizer)

    # Train
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(bert_config),
    )

    trainer.train()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
