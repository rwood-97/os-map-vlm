#!/bin/bash
#SBATCH --job-name=classify_mapreader_railspace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

echo "=== MapReader railspace classification (128px sub-crops of our 512px patches) ==="
uv run --no-sync python scripts/8b-classify_mapreader_symbols.py \
    --patches-dir data/patches_6inch_2nd_ed \
    --label railspace \
    --output data/patches_6inch_2nd_ed/mapreader_railspace.jsonl \
    --batch-size 16 \
    --num-workers 8
