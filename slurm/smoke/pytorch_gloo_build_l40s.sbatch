#!/bin/bash
#SBATCH --job-name=hpc-ai-torch-gloo
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_l40s
#SBATCH --output=slurm/pytorch-gloo-build-%j.out
#SBATCH --error=slurm/pytorch-gloo-build-%j.err

# Building does not require the GPU, but allocating on L40S lets the same job
# validate CUDA, NCCL, and Gloo before publishing the private wheel.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
cd "$PROJECT_ROOT"
bash slurm/setup_pytorch_gloo_environment.sh --project-root "$PROJECT_ROOT"
