#!/bin/bash
#SBATCH --job-name=mae_train_6inch_2nd_ed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

# ViT-B, pixel reconstruction target, from scratch, 6-inch 2nd ed. only.
echo "=== MAE training from scratch — ViT-B, 6-inch 2nd ed. only ==="
uv run --no-sync python ./scripts/6-train_mae.py \
    --shard-dirs data/shards_6inch_2nd_ed \
    --output-dir data/checkpoints/mae_6inch_2nd_ed \
    --epochs 20 \
    --shardshuffle 447 \
    --wandb-project os-map-vlm \
    --wandb-entity rosie-wood-the-alan-turing-institute \
    --wandb-run-name mae_scratch_6inch_2nd_ed \
    --compile \
    --num-workers 8
