#!/bin/bash
#SBATCH --job-name=h100-recovery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_h100
#SBATCH --output=slurm/h100-recovery-%j.out
#SBATCH --error=slurm/h100-recovery-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
# Do not activate the old virtualenv: its interpreter path is inaccessible.
python scripts/run_recovered_vllm.py scripts/smoke_recovered_distributed.py \
  --output "build/runtime_recovery/h100/job-${SLURM_JOB_ID}/REPORT.json"
