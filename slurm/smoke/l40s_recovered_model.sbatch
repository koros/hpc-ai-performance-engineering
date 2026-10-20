#!/bin/bash
#SBATCH --job-name=l40s-model-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --partition=gpu_l40s
#SBATCH --output=slurm/l40s-model-check-%j.out
#SBATCH --error=slurm/l40s-model-check-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
source "$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo-recovered/bin/activate_recovered_vllm"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/smoke_recovered_vllm_model.py \
  --output "build/runtime_recovery/l40s/model-job-${SLURM_JOB_ID}/REPORT.json"
