#!/bin/bash
#SBATCH --job-name=hpc-ai-vllm-build
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_l40s
#SBATCH --output=slurm/vllm-build-%j.out
#SBATCH --error=slurm/vllm-build-%j.err

# Build the private vLLM environment on the same Lovelace software stack used
# for the serving experiments. This script changes only the external paths
# configured by HPC_AI_VLLM_ENV and HPC_AI_VLLM_SOURCE.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
cd "$PROJECT_ROOT"
bash slurm/setup_vllm_environment.sh --project-root "$PROJECT_ROOT"
