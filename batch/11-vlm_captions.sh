#!/bin/bash
#SBATCH --job-name=vlm_captions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

echo "=== VLM captions — Source B ==="
uv run --no-sync python scripts/11-vlm_captions.py \
    --captions data/patches_6inch_2nd_ed/captions.jsonl \
    --patches-dir data/patches_6inch_2nd_ed \
    --output data/patches_6inch_2nd_ed/vlm_captions.jsonl \
    --model Qwen/Qwen3-VL-7B-Instruct \
    --batch-size 8
