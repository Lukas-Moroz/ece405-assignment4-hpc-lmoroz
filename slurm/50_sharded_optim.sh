#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --job-name=a4_sharded
#SBATCH --output=logs/sharded_%j.log
#SBATCH --error=logs/sharded_%j.log
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16

# Optimizer state sharding benchmarks (Problem optimizer_state_sharding_accounting).

source slurm/00_setup.sh
OUT=results/sharded_opt
mkdir -p "$OUT"

for SIZE in xl medium; do
    for MODE in non_sharded sharded; do
        python -m cs336_systems.benchmark_sharded \
            --backend nccl --world-size 2 --mode $MODE --size $SIZE \
            --context-length 128 --batch-size 4 \
            --n-warmup 2 --n-iters 5 \
            2>&1 | tee -a "$OUT/${SIZE}.log" || true
    done
done

echo "Done: $(date -Iseconds)"
