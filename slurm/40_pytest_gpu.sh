#!/bin/bash
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:NV-H100:1
#SBATCH --job-name=a4_pytest_gpu
#SBATCH --output=logs/pytest_gpu_%j.log
#SBATCH --error=logs/pytest_gpu_%j.log
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8

# Run the full pytest suite on a GPU node so the Triton tests are exercised.

source slurm/00_setup.sh

python -m pytest tests/ -v --junitxml=test_results.xml | tee logs/pytest_full.log

echo "Done: $(date -Iseconds)"
