#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:NV-RTX2080Ti:4
#SBATCH --job-name=a4_nccl_w4
#SBATCH --output=logs/nccl_w4_v2_%j.log
#SBATCH --error=logs/nccl_w4_v2_%j.log
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8

source slurm/00_setup.sh

python -m cs336_systems.benchmark_allreduce --backend nccl --world-size 4 \
    | tee results/benchmark_dist/allreduce_nccl_w4.csv

echo "Done: $(date -Iseconds)"
