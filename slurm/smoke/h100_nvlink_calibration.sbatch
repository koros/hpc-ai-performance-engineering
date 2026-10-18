#!/bin/bash
#SBATCH --job-name=h100-nvlink-calibration
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_h100
#SBATCH --output=slurm/h100-nvlink-calibration-%j.out
#SBATCH --error=slurm/h100-nvlink-calibration-%j.err

# Active diagnostic: two seconds of peer copies in each direction, plus idle/readout.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/calibrate_h100_nvlink.py \
  --output-dir "build/nvlink_calibration/h100/job-${SLURM_JOB_ID}"
