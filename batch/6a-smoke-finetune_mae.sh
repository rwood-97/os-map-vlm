#!/bin/bash
#SBATCH --job-name=mae_smoke_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env

echo "=== MAE smoke test (pretrained ViT-B/16 fine-tune) ==="
uv run --no-sync python ./scripts/6-train_mae.py \
    --shard-dirs data/shards_25inch data/shards_6inch_1st_ed \
                 data/shards_town_plans_1056 data/shards_town_plans_500 \
    --output-dir data/checkpoints/mae_smoke_finetune \
    --pretrained-encoder vit_base_patch16_224 \
    --max-steps 1000 \
    --warmup-steps 20 \
    --wandb-project os-map-vlm \
    --wandb-entity rosie-wood-the-alan-turing-institute \
    --wandb-run-name smoke_finetune_vit_b16 \
    --batch-size 256 \
    --num-workers 8
