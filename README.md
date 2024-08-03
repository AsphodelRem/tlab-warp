# tlab-warp
Codebase for my implementation of the WARP algorithm

[Original paper](https://arxiv.org/pdf/2406.16768)

## Project structure
- configs/config.toml - config file with hyperparameters
- warp.py - WARP implementation
- warp_train.py - script for running alignment process
- reward_model_train.py - script for training a reward model
- inference.py - script for running inference of aligned model
- dataset_utils.py - some additional functional for working with dataset
- conifg_utils - some additional functions for working with c toml config

## How to run
```
git clone https://github.com/AsphodelRem/tlab-warp
cd tlab-warp

# Train reward model
python3 reward_model_train.py --config configs/config.toml

# Train (align) SFT model
python3 warp_train.py --config configs/config.toml

# Inference (set user_input if you want to write your prompts)
python3 inference.py --path_to_align_model weights/aligned_model --config configs/config.toml --user_input True
```