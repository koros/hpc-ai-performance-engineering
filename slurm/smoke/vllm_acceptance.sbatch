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
HPC_AI_VLLM_ENV="${HPC_AI_VLLM_ENV:-${HOME}/.local/share/hpc-ai-performance-engineering/vllm-${VLLM_VERSION}-gloo}"

cd "$PROJECT_ROOT"
source slurm/modules_for_hpc_ai

if [[ ! -r "$HPC_AI_VLLM_ENV/bin/activate" ]]; then
  echo "ERROR: isolated vLLM environment is missing: $HPC_AI_VLLM_ENV" >&2
  echo "Run sbatch slurm/smoke/vllm_build_l40s.sbatch first." >&2
  exit 3
fi
unset PYTHONPATH PYTHONHOME
source "$HPC_AI_VLLM_ENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/src"
export TOKENIZERS_PARALLELISM=false

python scripts/check_private_torch.py \
  --expected-environment "$HPC_AI_VLLM_ENV" \
  --expected-version 2.8.0 \
  --expected-cuda 12.4 \
  --require-gloo \
  --require-nccl \
  --require-cuda \
  --smoke-process-groups

python -m hpc_ai_perf.cli check-runtime \
  --expected-gpu-model "$EXPECTED_GPU_MODEL" \
  --expected-torch-version 2.8.0 \
  --expected-torch-cuda 12.4 \
  --expected-gpus 1 \
  --require-vllm

ACCEPTANCE_OUTPUT_DIR="build/vllm_acceptance/$(tr '[:upper:]' '[:lower:]' <<<"$EXPECTED_GPU_MODEL")"
export HPC_AI_VLLM_ACCEPTANCE_RESULT_DIR="$ACCEPTANCE_OUTPUT_DIR/E006"

python -m hpc_ai_perf.cli run \
  --config configs/environment_validation/E006_vllm_1gpu.json \
  --output-dir "$ACCEPTANCE_OUTPUT_DIR" \
  --repetition 1

# The benchmark runner preserves failed attempts as JSON evidence and therefore
# does not make the CLI process fail. Convert the current job's E006 result into
# a strict Slurm gate so sacct cannot report COMPLETED for a failed acceptance.
python - <<'PY'
import json
import os
from pathlib import Path

result_dir = Path(os.environ["HPC_AI_VLLM_ACCEPTANCE_RESULT_DIR"])
job_id = os.environ["SLURM_JOB_ID"]
matches = []
for path in result_dir.glob("*.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    if str(record.get("environment", {}).get("slurm_job_id")) == job_id:
        matches.append((path, record))

if len(matches) != 1:
    raise SystemExit(f"Expected one E006 result for Slurm job {job_id}; found {len(matches)}")

path, record = matches[0]
metrics = record.get("metrics", {})
issues = []
if record.get("status") != "completed":
    issues.append(f"status is {record.get('status')!r}")
if record.get("error") is not None:
    issues.append(f"error is {record.get('error')}")
for key in ("vllm_import", "engine_start", "request_served"):
    if metrics.get(key) is not True:
        issues.append(f"{key} is not true")
if int(metrics.get("generated_tokens") or 0) < 1:
    issues.append("generated_tokens is not positive")
if float(metrics.get("throughput_tokens_sec") or 0.0) <= 0:
    issues.append("throughput_tokens_sec is not positive")

if issues:
    raise SystemExit(f"E006 acceptance failed for {path}: {'; '.join(issues)}")
print(f"E006 acceptance passed: {path}")
PY
