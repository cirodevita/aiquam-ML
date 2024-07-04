#!/bin/bash
#SBATCH --job-name=KNN
#SBATCH --output=logs/KNN.out
#SBATCH --error=logs/KNN.err
#SBATCH --partition=xhicpu

. venv/bin/activate

python -u main.py --is_training 1 --model KNN --data UEA --root_path ./dataset/AIQUAM/