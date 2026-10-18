#!/bin/bash
#SBATCH --job-name=h100-nvlink-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_h100
#SBATCH --output=slurm/h100-nvlink-probe-%j.out
#SBATCH --error=slurm/h100-nvlink-probe-%j.err

# Capability diagnostic only; no model download, benchmark load or service changes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/probe_h100_nvlink.py \
  --expected-gpus 2 \
  --output-dir "build/nvlink_probe/h100/job-${SLURM_JOB_ID}"
