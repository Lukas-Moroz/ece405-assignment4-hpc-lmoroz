#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:NV-H100:1
#SBATCH --job-name=a4_nsys
#SBATCH --output=logs/nsys_%j.log
#SBATCH --error=logs/nsys_%j.log
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8

# Nsight Systems profiles (Problem nsys_profile).

source slurm/00_setup.sh
OUT=results/nsys
mkdir -p "$OUT"

# Best-effort: nsys may not be on PATH; skip with a clear message if so.
if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found on PATH; skipping nsys profiles. Load Nsight Systems module first."
    exit 0
fi

# Profile a forward pass and a full training step at a few sizes / contexts.
for SIZE in small medium large; do
    for CTX in 128 512 1024; do
        nsys profile --pytorch=autograd-nvtx -o "$OUT/${SIZE}_ctx${CTX}_fwd" \
            python -m cs336_systems.benchmark_model --size $SIZE --context-length $CTX \
                --mode forward --warmup 2 --steps 3 --nvtx || true
        nsys profile --pytorch=autograd-nvtx -o "$OUT/${SIZE}_ctx${CTX}_train" \
            python -m cs336_systems.benchmark_model --size $SIZE --context-length $CTX \
                --mode train --warmup 2 --steps 3 --nvtx || true
    done
done

echo "Done: $(date -Iseconds)"
