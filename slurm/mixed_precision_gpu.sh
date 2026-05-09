#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=a4_mp
#SBATCH --output=logs/mixed_precision_%j.log
#SBATCH --error=logs/mixed_precision_%j.log
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4

source slurm/00_setup.sh
cd cs336-systems

echo "--- FP16 ---"
python -m cs336_systems.toy_mixed_precision --dtype fp16

echo "--- BF16 ---"
python -m cs336_systems.toy_mixed_precision --dtype bf16

echo "Done: $(date -Iseconds)"
