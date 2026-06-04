#!/bin/bash
#SBATCH --job-name=download_town_plans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

uv run --no-sync python ./scripts/4-download_town_plans_500.py
