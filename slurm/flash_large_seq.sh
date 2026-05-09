#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=a4_flash_large
#SBATCH --output=logs/flash_large_%j.log
#SBATCH --error=logs/flash_large_%j.log
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8

source slurm/00_setup.sh
cd cs336-systems

python -m cs336_systems.benchmark_attention \
    --implementations flash_triton \
    --dtypes bf16 fp32 \
    --seq-lengths 128 32768 65536 \
    --head-dims 16 32 64 128 \
    --n-warmup 3 --n-iters 20 2>&1

echo "Done: $(date -Iseconds)"
