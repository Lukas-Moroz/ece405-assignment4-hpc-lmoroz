#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --job-name=a4_bench_dist
#SBATCH --output=logs/bench_dist_%j.log
#SBATCH --error=logs/bench_dist_%j.log
#SBATCH --mem=256G
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16

# Distributed benchmarks (Problems distributed_communication_single_node,
# naive_ddp_benchmarking, minimal_ddp_flat_benchmarking,
# ddp_overlap_individual_parameters_benchmarking, ddp_bucketed_benchmarking).
# Koa cannot do multi-node, so we run single-node multi-GPU only.

source slurm/00_setup.sh
OUT=results/benchmark_dist
mkdir -p "$OUT"

echo "=== all-reduce: NCCL on GPUs ==="
for W in 2; do
    python -m cs336_systems.benchmark_allreduce --backend nccl --world-size $W \
        | tee "$OUT/allreduce_nccl_w${W}.csv" || true
done

echo "=== all-reduce: Gloo on CPU ==="
for W in 2 4; do
    python -m cs336_systems.benchmark_allreduce --backend gloo --world-size $W \
        | tee "$OUT/allreduce_gloo_w${W}.csv" || true
done

# DDP variants on the XL model size, falling back to medium if XL OOMs.
for SIZE in xl medium small; do
    echo "=== DDP variants on $SIZE (1 node x 2 GPU) ==="
    for MODE in naive flat overlap; do
        python -m cs336_systems.benchmark_ddp --backend nccl --world-size 2 \
            --mode $MODE --size $SIZE --context-length 128 --batch-size 4 \
            --n-warmup 2 --n-iters 5 \
            2>&1 | tee -a "$OUT/ddp_${SIZE}.log" || true
    done
    for BUCKET in 1 10 100 1000; do
        python -m cs336_systems.benchmark_ddp --backend nccl --world-size 2 \
            --mode bucketed --size $SIZE --context-length 128 --batch-size 4 \
            --n-warmup 2 --n-iters 5 --bucket-mb $BUCKET \
            2>&1 | tee -a "$OUT/ddp_${SIZE}.log" || true
    done
    # Stop after first size that ran (to keep SLURM job time bounded).
    if grep -q "iter_ms=" "$OUT/ddp_${SIZE}.log" 2>/dev/null; then
        break
    fi
done

echo "Done: $(date -Iseconds)"
