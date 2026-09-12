#!/bin/bash
#SBATCH --job-name=hpc-ai-fp8-acceptance
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_l40s
#SBATCH --output=slurm/fp8-acceptance-%j.out
#SBATCH --error=slurm/fp8-acceptance-%j.err

# Diagnostic gate only: these smoke results are not canonical E086/E093 evidence.
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
EXPECTED_GPU_MODEL="${HPC_AI_EXPECTED_GPU_MODEL:-L40S}"
FP8_ENV="${HPC_AI_FP8_ENV:-${HOME}/.local/share/hpc-ai-performance-engineering/fp8-training-te2.8.0}"
PLATFORM_ID="$(tr '[:upper:]' '[:lower:]' <<<"$EXPECTED_GPU_MODEL")"
ACCEPTANCE_OUTPUT_DIR="$PROJECT_ROOT/build/fp8_acceptance/$PLATFORM_ID"

cd "$PROJECT_ROOT"
source slurm/modules_for_hpc_ai

if [[ ! -r "$FP8_ENV/bin/activate" ]]; then
  echo "ERROR: isolated FP8 training environment is missing: $FP8_ENV" >&2
  echo "Run make submit-fp8-training-build and wait for it to succeed first." >&2
  exit 3
fi
unset PYTHONPATH PYTHONHOME
source "$FP8_ENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false

python scripts/check_private_torch.py \
  --expected-environment "$FP8_ENV" \
  --expected-version 2.8.0 \
  --expected-cuda 12.4 \
  --require-gloo \
  --require-nccl \
  --require-cuda

python -m hpc_ai_perf.cli run \
  --config configs/smoke/fp8_gpt2_training_acceptance.json \
  --output-dir "$ACCEPTANCE_OUTPUT_DIR"
python -m hpc_ai_perf.cli run \
  --config configs/smoke/fp8_distilbert_training_acceptance.json \
  --output-dir "$ACCEPTANCE_OUTPUT_DIR"

python scripts/verify_fp8_training_acceptance.py \
  --result-root "$ACCEPTANCE_OUTPUT_DIR" \
  --expected-gpu-model "$EXPECTED_GPU_MODEL"
