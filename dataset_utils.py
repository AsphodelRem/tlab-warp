from tqdm import tqdm
from datasets import load_dataset


def get_datasets_for_warp(config):
    """
    Load datasets and prepare training and testing prompts.
    
    Args:
    - config (dict): Configuration dictionary containing dataset and warp parameters.
    
    Returns:
    - Tuple[list, list]: Two lists containing the training and testing prompts, respectively.
    """
    dataset = load_dataset(config["dataset"]["dataset_name"])

    max_length = config["warp"]["max_length"]
    max_test_samples = config["dataset"]["test_max_size"]

    train_prompts = [text[:max_length] for text in dataset["train"]["text"]]
    test_prompts = [text[:max_length] for text in dataset["test"]["text"][:max_test_samples]]

    return train_prompts, test_prompts

def get_datasets_for_reward_model(config, tokenizer):
    """
    Prepare datasets for training a reward model by tokenizing and creating pairs of samples.
    
    Args:
    - config (dict): Configuration dictionary containing dataset parameters.
    - tokenizer (PreTrainedTokenizer): Tokenizer instance to preprocess text data.
    
    Returns:
    - Tuple[list, list]: Two lists containing training and testing pairs for the reward model.
    """
    def _preprocess_function(data):
        encodings = tokenizer(
            data["text"], truncation=True, padding=True, return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

    def _create_pairs(pos, neg):
        pairs = []
        for pos_sample, neg_sample in zip(pos, neg):
            pairs.append(
                {
                    "input_ids_chosen": pos_sample["input_ids"],
                    "attention_mask_chosen": pos_sample["attention_mask"],
                    "input_ids_rejected": neg_sample["input_ids"],
                    "attention_mask_rejected": neg_sample["attention_mask"],
                }
            )
        return pairs

    dataset = load_dataset(config["dataset"]["dataset_name"])

    train_dataset = dataset["train"].map(_preprocess_function, batched=True)
    test_dataset = dataset["test"].map(_preprocess_function, batched=True)

    pos_train_dataset = train_dataset.filter(lambda x: x["label"] == 1)
    neg_train_dataset = train_dataset.filter(lambda x: x["label"] == 0)
    pos_test_dataset = test_dataset.filter(lambda x: x["label"] == 1)
    neg_test_dataset = test_dataset.filter(lambda x: x["label"] == 0)

    train_pairs = _create_pairs(pos_train_dataset, neg_train_dataset)
    test_pairs = _create_pairs(pos_test_dataset, neg_test_dataset)

    return train_pairs, test_pairs
