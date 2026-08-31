#!/bin/bash
#SBATCH --job-name=hpc-ai-vllm-acceptance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_l40s
#SBATCH --output=slurm/vllm-acceptance-%j.out
#SBATCH --error=slurm/vllm-acceptance-%j.err

# This is an environment gate, not canonical matrix evidence. It must pass
# before E006 and the remaining vLLM rows are re-enabled in the platform matrix.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
EXPECTED_GPU_MODEL="${HPC_AI_EXPECTED_GPU_MODEL:-L40S}"
VLLM_VERSION="${HPC_AI_VLLM_VERSION:-0.10.2}"
HPC_AI_VLLM_ENV="${HPC_AI_VLLM_ENV:-${HOME}/.local/share/hpc-ai-performance-engineering/vllm-${VLLM_VERSION}}"

cd "$PROJECT_ROOT"
source slurm/modules_for_hpc_ai

if [[ ! -r "$HPC_AI_VLLM_ENV/bin/activate" ]]; then
  echo "ERROR: isolated vLLM environment is missing: $HPC_AI_VLLM_ENV" >&2
  echo "Run sbatch slurm/smoke/vllm_build_l40s.sbatch first." >&2
  exit 3
fi
source "$HPC_AI_VLLM_ENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python -m hpc_ai_perf.cli check-runtime \
  --expected-gpu-model "$EXPECTED_GPU_MODEL" \
  --expected-torch-version 2.8.0 \
  --expected-torch-cuda 12.4 \
  --expected-gpus 1 \
  --require-vllm

python -m hpc_ai_perf.cli run \
  --config configs/environment_validation/E006_vllm_1gpu.json \
  --output-dir build/vllm_acceptance/"$(tr '[:upper:]' '[:lower:]' <<<"$EXPECTED_GPU_MODEL")" \
  --repetition 1
