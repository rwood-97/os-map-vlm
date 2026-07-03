#!/bin/bash
# Worker script: start Ray on each node.
# Worker nodes (SLURM_NODEID > 0) block in Ray; only rank 0 runs the Python script.
# Called via srun from batch/11-vlm_captions.sh and batch/11a-vlm_captions_smoke.sh.
set -e

export VLLM_HOST_IP=$(hostname -i)

if [[ "$SLURM_NODEID" -eq 0 && "$SLURM_LOCALID" -eq 0 ]]; then
    ray start --head --node-ip-address "$PRIMARY_IP" --port "$PRIMARY_PORT"
elif [[ "$SLURM_LOCALID" -eq 0 ]]; then
    sleep 10
    ray start --block --address "$PRIMARY_IP:$PRIMARY_PORT" --node-ip-address "$VLLM_HOST_IP"
fi

sleep 20  # let the Ray cluster fully form
ray status
uv run --no-sync python scripts/11-vlm_captions.py "$@"
