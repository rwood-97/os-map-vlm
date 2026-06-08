#!/bin/bash
#SBATCH --job-name=mae_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

uv run --no-sync python ./scripts/6-train_mae.py \
    --shard-dirs data/shards_25inch data/shards_6inch_1st_ed \
                 data/shards_town_plans_1056 data/shards_town_plans_500 \
    --output-dir data/checkpoints/mae_full \
    --epochs 20 \
    --compile
