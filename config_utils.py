import toml

def create_default_config():
    config = {
        'reward_model_trainer': {
            'model_name_or_path': 'distilbert/distilbert-base-cased',
            'output_dir': 'd_bert',
            'per_device_train_batch_size': 64,
            'num_train_epochs': 1,
            'gradient_checkpointing': True,
            'remove_unused_columns': False,
            'max_length': 512,
        },
        'warp': {
            'sft_model_name_or_path': 'lvwerra/gpt2-imdb', 
            'reward_model': 'd_bert/checkpoint-196', 
            'output_dir': 'aligned_model',
            'iterations': 2, 
            'ml_runs': 2, 
            'training_steps': 100, 
            'ema_update_rate': 0.05, 
            'liti_update_rate': 0.05,
            'beta': 0.25, 
            'batch_size': 64,
            'lr': 1e-6,
            'max_length': 20
        },
        'dataset': {
            'dataset_name': 'stanfordnlp/imdb',
            'test_max_size': 100,
        }
    }
    return config

def save_config(config, filename):
    with open(filename, 'w') as f:
        toml.dump(config, f)

def load_config(filename):
    with open(filename, 'r') as f:
        config = toml.load(f)
    return config