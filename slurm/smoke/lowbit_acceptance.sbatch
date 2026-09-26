#!/bin/bash
#SBATCH --job-name=hpc-ai-lowbit-acceptance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm/lowbit-acceptance-%j.out
#SBATCH --error=slurm/lowbit-acceptance-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source slurm/modules_for_hpc_ai
unset PYTHONPATH PYTHONHOME
source "${HPC_AI_LOWBIT_ENV:?}/bin/activate"
export PYTHONPATH="$PWD/src"
export VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NVML_BASED_CUDA_CHECK=1
LOWBIT_EXTRA=()
if [[ "${HPC_AI_LOWBIT_INCLUDE_BNB:-0}" == "1" ]]; then
    LOWBIT_EXTRA+=(--include-bnb)
fi
python scripts/lowbit_acceptance.py run \
    --platform "${HPC_AI_LOWBIT_PLATFORM:?}" \
    --wave "${HPC_AI_LOWBIT_WAVE:-lowbit-serving}" \
    --result-root "build/lowbit_acceptance/${HPC_AI_LOWBIT_PLATFORM}/${SLURM_JOB_ID}" \
    "${LOWBIT_EXTRA[@]}"
