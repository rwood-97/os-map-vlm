#!/bin/bash
#SBATCH --job-name=mae_finetune_6inch_2nd_ed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

# ViT-B initialised from ImageNet-pretrained weights, pixel reconstruction target,
# 6-inch 2nd ed. only.
# 100 epochs (up from 10) -- this run was already converging much faster per step
# than the from-scratch run (see probe results), so it's the more promising of the
# two to keep investing compute in.
# This script auto-resumes from the latest checkpoint in --output-dir, so if a
# single job doesn't reach --epochs within the walltime, just resubmit it again.
echo "=== MAE fine-tune from pretrained ViT-B/16 — 6-inch 2nd ed. only ==="
uv run --no-sync python ./scripts/6-train_mae.py \
    --shard-dirs data/shards_6inch_2nd_ed \
    --output-dir data/checkpoints/mae_finetune_6inch_2nd_ed \
    --pretrained-encoder vit_base_patch16_224 \
    --epochs 100 \
    --shardshuffle 447 \
    --wandb-project os-map-vlm \
    --wandb-entity rosie-wood-the-alan-turing-institute \
    --wandb-run-name finetune_vit_b16_6inch_2nd_ed \
    --compile \
    --num-workers 8
