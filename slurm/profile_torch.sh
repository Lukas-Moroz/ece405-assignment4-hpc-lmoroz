#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=a4_profile
#SBATCH --output=logs/profile_torch_%j.log
#SBATCH --error=logs/profile_torch_%j.log
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8

source slurm/00_setup.sh
mkdir -p results/profile_torch

for SIZE in small medium large; do
    for MODE in forward fwd_bwd train; do
        python -m cs336_systems.profile_torch \
            --size $SIZE --context-length 256 --mode $MODE \
            --warmup 3 --active 3 --top-n 30 2>&1 || true
        echo "---"
    done
done

echo "Done: $(date -Iseconds)"
