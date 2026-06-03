#!/bin/bash
#SBATCH --job-name=create_shards
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Create WebDataset shards from a patchified map series.
#
# Usage:
#   sbatch batch/11-create_shards.sh <series> [max_patches]
#
# Examples:
#   sbatch batch/11-create_shards.sh town_plans_500
#   sbatch batch/11-create_shards.sh 6inch_2nd_ed 500000   # smoke-test subset
#
# Valid series names: town_plans_500, town_plans_1056, 25inch, 6inch_1st_ed, 6inch_2nd_ed
# No GPU needed — this is a CPU I/O job.

module load gcc-native/12.3

SERIES=${1:?Usage: sbatch 11-create_shards.sh <series> [max_patches]}
PATCHES_DIR="data/patches_${SERIES}"
OUTPUT_DIR="data/shards_${SERIES}"
MAX_PATCHES=${2:-}

CMD="uv run --no-sync python ./scripts/11-create_shards.py \
    --series ${SERIES} \
    --patches-dir ${PATCHES_DIR} \
    --output-dir ${OUTPUT_DIR}"

if [ -n "${MAX_PATCHES}" ]; then
    CMD="${CMD} --max-patches ${MAX_PATCHES}"
fi

echo "Series:      ${SERIES}"
echo "Patches dir: ${PATCHES_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Max patches: ${MAX_PATCHES:-all}"
echo ""

eval ${CMD}
