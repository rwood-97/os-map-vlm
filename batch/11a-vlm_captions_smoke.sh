#!/bin/bash
#SBATCH --job-name=vlm_captions_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

echo "=== VLM captions smoke test (20 patches) ==="
uv run --no-sync python scripts/11-vlm_captions.py \
    --captions data/patches_6inch_2nd_ed/captions.jsonl \
    --patches-dir data/patches_6inch_2nd_ed \
    --output data/patches_6inch_2nd_ed/vlm_captions_smoke.jsonl \
    --model Qwen/Qwen3-VL-7B-Instruct \
    --batch-size 4 \
    --max-samples 20
