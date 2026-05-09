#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=a4_warmup
#SBATCH --output=logs/warmup_sweep_%j.log
#SBATCH --error=logs/warmup_sweep_%j.log
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8

source slurm/00_setup.sh

for WARMUP in 0 1 2 5; do
  for SIZE in small medium; do
    python -m cs336_systems.benchmark_model --size $SIZE --context-length 256 \
      --mode fwd_bwd --warmup $WARMUP --steps 10 --dtype fp32 2>&1
    echo "---"
  done
done

echo "Done: $(date -Iseconds)"
