#!/bin/bash
# Common setup snippet sourced by every sbatch script.
# Sets up paths and activates the uv-managed virtual environment.

set -euo pipefail

REPO=/mnt/lustre/koa/scratch/lmoroz/ece405-assignment4-hpc-lmoroz
cd "$REPO"

# Lazily build the venv if it doesn't exist.
if [ ! -d ".venv" ]; then
    /home/lmoroz/.local/bin/uv sync --no-dev
fi

# Re-link the editable install in case the project layout (e.g. module-root)
# changed since the venv was last synced.  --no-deps avoids touching torch,
# which would otherwise trigger a long re-install on every job.
/home/lmoroz/.local/bin/uv pip install -e . --no-deps --quiet

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"

echo "============================================================"
echo "Job:    ${SLURM_JOB_NAME:-?} (id ${SLURM_JOB_ID:-?})"
echo "Node:   ${SLURMD_NODENAME:-?}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
echo "Start:  $(date -Iseconds)"
echo "============================================================"
