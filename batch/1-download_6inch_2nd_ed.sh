#!/bin/bash
#SBATCH --job-name=download_6in_2nd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync python ./scripts/1-download_6inch_2nd_ed.py
