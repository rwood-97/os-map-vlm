#!/bin/bash
#SBATCH --job-name=download_25inch_scot2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync python ./scripts/5b-download_25inch_scotland2.py
