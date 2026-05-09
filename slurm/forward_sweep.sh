#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=a4_fwd_sweep
#SBATCH --output=logs/fwd_sweep_%j.log
#SBATCH --error=logs/fwd_sweep_%j.log
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8

source slurm/00_setup.sh

for SIZE in small medium large xl; do
  for CTX in 128 256 512 1024; do
    python -m cs336_systems.benchmark_model --size $SIZE --context-length $CTX \
      --mode forward --warmup 5 --steps 10 --dtype fp32 2>&1 || true
  done
done

for CTX in 128 256 512; do
  python -m cs336_systems.benchmark_model --size 2.7B --context-length $CTX \
    --mode forward --warmup 5 --steps 10 --dtype fp32 2>&1 || true
done

echo "Done: $(date -Iseconds)"
