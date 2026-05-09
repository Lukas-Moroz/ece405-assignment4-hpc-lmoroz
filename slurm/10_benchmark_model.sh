#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:NV-H100:1
#SBATCH --job-name=a4_bench_model
#SBATCH --output=logs/bench_model_%j.log
#SBATCH --error=logs/bench_model_%j.log
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8

# End-to-end Transformer benchmarks (Problems benchmarking_script,
# benchmarking_mixed_precision, memory_profiling, torch_compile (b)).

source slurm/00_setup.sh

OUT=results/benchmark_model
mkdir -p "$OUT"

echo "--- forward+backward, FP32, all sizes / context lengths ---"
for SIZE in small medium large xl 2.7B; do
    for CTX in 128 256 512 1024; do
        python -m cs336_systems.benchmark_model --size $SIZE --context-length $CTX \
            --mode fwd_bwd --warmup 5 --steps 10 \
            2>&1 | tee -a "$OUT/fp32_fwdbwd.log" || true
    done
done

echo "--- mixed precision (BF16) ---"
for SIZE in small medium large xl 2.7B; do
    for CTX in 128 256 512 1024; do
        python -m cs336_systems.benchmark_model --size $SIZE --context-length $CTX \
            --mode fwd_bwd --dtype bf16 --warmup 5 --steps 10 \
            2>&1 | tee -a "$OUT/bf16_fwdbwd.log" || true
    done
done

echo "--- torch.compile (whole model) ---"
for SIZE in small medium large; do
    python -m cs336_systems.benchmark_model --size $SIZE --context-length 256 \
        --mode train --warmup 5 --steps 10 --compile \
        2>&1 | tee -a "$OUT/compile.log" || true
done

echo "--- memory profile (2.7B forward / training) ---"
for CTX in 128 256 512; do
    python -m cs336_systems.benchmark_model --size 2.7B --context-length $CTX \
        --mode forward --warmup 2 --steps 3 \
        --memory-snapshot "$OUT/mem_2_7B_ctx${CTX}_fwd.pickle" 2>&1 | tee -a "$OUT/memory.log" || true
    python -m cs336_systems.benchmark_model --size 2.7B --context-length $CTX \
        --mode train --warmup 2 --steps 3 \
        --memory-snapshot "$OUT/mem_2_7B_ctx${CTX}_train.pickle" 2>&1 | tee -a "$OUT/memory.log" || true
done

echo "Done: $(date -Iseconds)"
