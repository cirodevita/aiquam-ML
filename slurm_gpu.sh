#!/bin/bash
#SBATCH --job-name=TimesNet
#SBATCH --output=logs/TimesNet.out
#SBATCH --error=logs/TimesNet.err
#SBATCH --partition=xgpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla:1

. venv/bin/activate

python -u main.py \
    --is_training 1 \
    --model TimesNet \
    --data UEA \
    --root_path ./dataset/AIQUAM/ \
    --itr 1 \
    --train_epochs 100 \
    --batch_size 16 \
    --patience 10 \
    --learning_rate 0.001 \
    --e_layers 3 \
    --top_k 3 \
    --d_model 128 \
    --d_ff 256
