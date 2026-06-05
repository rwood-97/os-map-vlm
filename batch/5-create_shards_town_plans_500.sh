#!/bin/bash
#SBATCH --job-name=create_shards_town_plans_500
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc-native/12.3

uv run --no-sync python ./scripts/5-create_shards.py \
    --series town_plans_500 \
    --patches-dir data/patches_town_plans_500 \
    --output-dir data/shards_town_plans_500
