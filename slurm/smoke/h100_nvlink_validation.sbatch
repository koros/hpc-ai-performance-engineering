#!/bin/bash
#SBATCH --job-name=h100-nvlink-validation
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_h100
#SBATCH --array=0-1%1
#SBATCH --output=slurm/h100-nvlink-validation-%A_%a.out
#SBATCH --error=slurm/h100-nvlink-validation-%A_%a.err

# Two new payload sizes, serialized; the completed 64 MiB calibration is reused.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PAYLOAD_SIZES=(16 128)
python scripts/calibrate_h100_nvlink.py \
  --payload-mib "${PAYLOAD_SIZES[SLURM_ARRAY_TASK_ID]}" \
  --output-dir "build/nvlink_calibration/h100/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
