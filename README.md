# tlab-warp
Codebase for my implementation of the WARP algorithm

[Original paper](https://arxiv.org/pdf/2406.16768)

## Project structure
- configs/config.toml - config file with hyperparameters
- reports/report_rus.pdf - experiments details
- warp.py - WARP implementation
- warp_train.py - script for running alignment process
- reward_model_train.py - script for training a reward model
- inference.py - script for running inference of aligned model
- dataset_utils.py - some additional functional for working with dataset
- conifg_utils - some additional functions for working with c toml config
- hyperparameters_experiment.py - experiment with number of training steps
- comparing_aligned_with_sft.py - comparing with non-aligned model

## How to run
```
git clone https://github.com/AsphodelRem/tlab-warp
cd tlab-warp

# If you want to create virtual env (on Linux)
python3 -m venv venv
. venv/bin/activate

pip3 install -r requirements.txt 

# Train reward model
python3 reward_model_train.py --config configs/config.toml

# Set path to saved reward model in config.toml if nesessary ([warp][reward_model])

# Train (align) SFT model
python3 warp_train.py --config configs/config.toml

# Inference (set user_input if you want to write your prompts)
python3 inference.py --path_to_align_model weights/aligned_model --config configs/config.toml --user_input True

# Run experiments
python3 hyperparameters_experiment.py --config configs/config.toml 
python3 comparing_aligned_with_sft.py --path_to_aligned_model weights/aligned_model --config configs/config.toml 
```
