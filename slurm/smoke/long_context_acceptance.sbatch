#!/bin/bash
#SBATCH --job-name=hpc-ai-long-context-acceptance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm/long-context-acceptance-%j.out
#SBATCH --error=slurm/long-context-acceptance-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
unset PYTHONPATH PYTHONHOME
source "${HPC_AI_LONG_CONTEXT_ENV:?}/bin/activate"
export PYTHONPATH="$PWD/src"
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NVML_BASED_CUDA_CHECK=1
python scripts/long_context_acceptance.py run \
    --platform "${HPC_AI_LONG_CONTEXT_PLATFORM:?}" \
    --result-root "build/long_context_acceptance/${HPC_AI_LONG_CONTEXT_PLATFORM}/${SLURM_JOB_ID}"
