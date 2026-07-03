#!/bin/bash
#SBATCH --job-name=vlm_captions
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=72
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load brics/default
module load cuda/12.6
module load gcc-native/12.3
module load brics/nccl

source .env

export PRIMARY_PORT=$((30000 + SLURM_JOB_ID % 16384))
export PRIMARY_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export PRIMARY_IP=$(srun --nodes=1 --ntasks=1 -w "$PRIMARY_HOST" hostname -i | tr -d ' ')
echo "Head node: $PRIMARY_HOST ($PRIMARY_IP:$PRIMARY_PORT)"

echo "=== VLM captions — Source B 235B, 2 nodes ==="
srun -N${SLURM_NNODES} -n${SLURM_NNODES} -l scripts/vllm_run_captions.sh \
    --captions data/patches_6inch_2nd_ed/captions.jsonl \
    --patches-dir data/patches_6inch_2nd_ed \
    --output data/patches_6inch_2nd_ed/vlm_captions.jsonl \
    --intermediate data/patches_6inch_2nd_ed/vlm_captions_quadrants.jsonl \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct
wait
