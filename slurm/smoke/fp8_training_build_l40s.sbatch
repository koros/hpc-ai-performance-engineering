#!/bin/bash
#SBATCH --job-name=hpc-ai-fp8-build
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_l40s
#SBATCH --output=slurm/fp8-build-%j.out
#SBATCH --error=slurm/fp8-build-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
cd "$PROJECT_ROOT"
if [[ -z "${HPC_AI_NCCL_HOME:-}" ]]; then
  echo "ERROR: HPC_AI_NCCL_HOME must name a matching NCCL development root" >&2
  exit 2
fi
bash slurm/setup_fp8_training_environment.sh \
  --project-root "$PROJECT_ROOT" \
  --nccl-home "$HPC_AI_NCCL_HOME"
