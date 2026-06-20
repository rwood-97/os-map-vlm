#!/bin/bash
#SBATCH --job-name=mae_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load cuda/12.6
module load gcc-native/12.3

source .env 

# Create shards for each series that has patches (skip if already done)
for series in 25inch 6inch_1st_ed town_plans_1056 town_plans_500; do
    if [ ! -d "data/shards_${series}" ]; then
        echo "=== Creating shards: ${series} ==="
        uv run --no-sync python ./scripts/5-create_shards.py \
            --series "${series}" \
            --patches-dir "data/patches_${series}" \
            --output-dir "data/shards_${series}"
    else
        echo "=== Shards already exist: ${series}, skipping ==="
    fi
done

# Smoke test: 200 steps to confirm loss decreases and job completes cleanly
echo "=== MAE smoke test ==="
uv run --no-sync python ./scripts/6-train_mae.py \
    --shard-dirs data/shards_25inch data/shards_6inch_1st_ed \
                 data/shards_town_plans_1056 data/shards_town_plans_500 \
    --output-dir data/checkpoints/mae_smoke \
    --max-steps 1000 \
    --warmup-steps 20 \
    --wandb-project os-map-vlm \
    --wandb-entity rosie-wood-the-alan-turing-institute \
    --wandb-run-name smoke_bs_256 \
    --batch-size 256 \
    --num-workers 8
