#!/bin/bash
#SBATCH --job-name=create_shards_6inch_1st_ed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc-native/12.3

uv run --no-sync python ./scripts/7-create_shards.py \
    --series 6inch_1st_ed \
    --patches-dir data/patches_6inch_1st_ed \
    --output-dir data/shards_6inch_1st_ed
