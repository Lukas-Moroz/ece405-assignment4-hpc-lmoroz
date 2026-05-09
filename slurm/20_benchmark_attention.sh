#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:NV-H100:1
#SBATCH --job-name=a4_bench_attn
#SBATCH --output=logs/bench_attn_%j.log
#SBATCH --error=logs/bench_attn_%j.log
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8

# Attention benchmarks (Problems pytorch_attention, torch_compile (a),
# flash_benchmarking).

source slurm/00_setup.sh
OUT=results/benchmark_attention
mkdir -p "$OUT"

echo "--- vanilla / compiled / flash_triton sweep, BF16 ---"
python -m cs336_systems.benchmark_attention \
    --implementations pytorch compiled flash_triton \
    --dtypes bf16 \
    --seq-lengths 256 1024 4096 8192 16384 \
    --head-dims 16 32 64 128 \
    --batch 1 --n-iters 50 --n-warmup 5 \
    | tee "$OUT/sweep_bf16.csv"

echo "--- FP32 sweep (smaller scope to avoid OOM) ---"
python -m cs336_systems.benchmark_attention \
    --implementations pytorch compiled flash_triton \
    --dtypes fp32 \
    --seq-lengths 256 1024 4096 8192 \
    --head-dims 16 32 64 128 \
    --batch 1 --n-iters 50 --n-warmup 5 \
    | tee "$OUT/sweep_fp32.csv"

echo "Done: $(date -Iseconds)"
