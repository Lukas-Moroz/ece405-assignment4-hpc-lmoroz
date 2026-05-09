#!/bin/bash
# Submit every Slurm job in a sensible order.
# IDs of dependent jobs are passed via --dependency=afterany.

set -euo pipefail

cd "$(dirname "$0")/.."

JID1=$(sbatch --parsable slurm/40_pytest_gpu.sh)
echo "submitted pytest_gpu       $JID1"

JID2=$(sbatch --parsable --dependency=afterany:$JID1 slurm/10_benchmark_model.sh)
echo "submitted benchmark_model  $JID2"

JID3=$(sbatch --parsable --dependency=afterany:$JID1 slurm/20_benchmark_attention.sh)
echo "submitted benchmark_attn   $JID3"

JID4=$(sbatch --parsable --dependency=afterany:$JID1 slurm/30_benchmark_dist.sh)
echo "submitted benchmark_dist   $JID4"

JID5=$(sbatch --parsable --dependency=afterany:$JID4 slurm/50_sharded_optim.sh)
echo "submitted sharded_optim    $JID5"

JID6=$(sbatch --parsable --dependency=afterany:$JID2 slurm/11_nsys_profile.sh)
echo "submitted nsys_profile     $JID6"
