#!/usr/bin/env bash
# Build an isolated Transformer Engine environment against the private Torch ABI.
set -euo pipefail

TE_VERSION="${HPC_AI_TE_VERSION:-2.8.0}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FP8_BASE="${HPC_AI_FP8_BASE:-${HOME}/.local/share/hpc-ai-performance-engineering}"
FP8_ENV="${HPC_AI_FP8_ENV:-${FP8_BASE}/fp8-training-te${TE_VERSION}}"
TORCH_WHEEL_DIR="${HPC_AI_TORCH_WHEEL_DIR:-${FP8_BASE}/wheels/pytorch-2.8.0-cu124-gloo}"
CUDNN_HOME="${HPC_AI_CUDNN_HOME:-}"
NCCL_HOME="${HPC_AI_NCCL_HOME:-}"
MAX_JOBS="${HPC_AI_TE_MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

usage() {
  cat <<'EOF'
Usage: bash slurm/setup_fp8_training_environment.sh [options]

Options:
  --project-root PATH     Repository containing slurm/modules_for_hpc_ai.
  --env-dir PATH          Destination virtual environment.
  --torch-wheel-dir PATH  Private PyTorch 2.8/CUDA 12.4 wheel directory.
  --cudnn-home PATH       Matching cuDNN development root containing include/cudnn.h.
  --nccl-home PATH        Matching NCCL development root containing include/nccl.h.
  --max-jobs COUNT        Parallel extension build jobs.
  --help                  Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --env-dir) FP8_ENV="$2"; shift 2 ;;
    --torch-wheel-dir) TORCH_WHEEL_DIR="$2"; shift 2 ;;
    --cudnn-home) CUDNN_HOME="$2"; shift 2 ;;
    --nccl-home) NCCL_HOME="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
FP8_ENV="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$FP8_ENV")"
TORCH_WHEEL_DIR="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$TORCH_WHEEL_DIR")"
if [[ -n "$CUDNN_HOME" ]]; then
  CUDNN_HOME="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$CUDNN_HOME")"
fi
if [[ -n "$NCCL_HOME" ]]; then
  NCCL_HOME="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$NCCL_HOME")"
fi
if [[ ! "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max-jobs must be a positive integer; got $MAX_JOBS" >&2
  exit 2
fi

# Strip a caller's active virtual environment before loading the cluster stack.
if [[ -n "${VIRTUAL_ENV:-}" && "$PATH" == "$VIRTUAL_ENV/bin:"* ]]; then
  export PATH="${PATH#"$VIRTUAL_ENV/bin:"}"
fi
unset VIRTUAL_ENV PYTHONHOME
hash -r

cd "$PROJECT_ROOT"
source slurm/modules_for_hpc_ai
if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME}/.local/bin/uv" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is unavailable; install it in the user account before retrying." >&2
  exit 3
fi

if [[ -z "$CUDNN_HOME" ]]; then
  echo "ERROR: set --cudnn-home or HPC_AI_CUDNN_HOME to a matching cuDNN development root" >&2
  exit 3
fi
if [[ ! -r "$CUDNN_HOME/include/cudnn.h" ]]; then
  echo "ERROR: cuDNN header is missing: $CUDNN_HOME/include/cudnn.h" >&2
  exit 3
fi
if ! compgen -G "$CUDNN_HOME/lib/libcudnn.so*" >/dev/null; then
  echo "ERROR: cuDNN shared library is missing under $CUDNN_HOME/lib" >&2
  exit 3
fi
if [[ -z "$NCCL_HOME" ]]; then
  echo "ERROR: set --nccl-home or HPC_AI_NCCL_HOME to a matching NCCL development root" >&2
  exit 3
fi
if [[ ! -r "$NCCL_HOME/include/nccl.h" ]]; then
  echo "ERROR: NCCL header is missing: $NCCL_HOME/include/nccl.h" >&2
  exit 3
fi
if ! compgen -G "$NCCL_HOME/lib/libnccl.so*" >/dev/null; then
  echo "ERROR: NCCL shared library is missing under $NCCL_HOME/lib" >&2
  exit 3
fi

BOOTSTRAP_PYTHON="$(command -v python3)"
unset PYTHONPATH PYTHONHOME

torch_wheels=("$TORCH_WHEEL_DIR"/torch-2.8.0-*.whl)
if [[ ${#torch_wheels[@]} -ne 1 || ! -e "${torch_wheels[0]}" ]]; then
  echo "ERROR: expected one private Torch 2.8.0 wheel in $TORCH_WHEEL_DIR" >&2
  exit 3
fi
TORCH_WHEEL="${torch_wheels[0]}"
if [[ ! -r "$TORCH_WHEEL_DIR/BUILD_MANIFEST.json" ]]; then
  echo "ERROR: private Torch build manifest is missing: $TORCH_WHEEL_DIR/BUILD_MANIFEST.json" >&2
  exit 3
fi

if [[ -e "$FP8_ENV" && ! -x "$FP8_ENV/bin/python" ]]; then
  echo "ERROR: FP8 environment path exists but is not a virtual environment: $FP8_ENV" >&2
  exit 3
fi
if [[ ! -e "$FP8_ENV" ]]; then
  mkdir -p "$(dirname "$FP8_ENV")"
  "$BOOTSTRAP_PYTHON" -m venv "$FP8_ENV"
fi

uv pip install --python "$FP8_ENV/bin/python" "$TORCH_WHEEL"
uv pip install --python "$FP8_ENV/bin/python" \
  "setuptools>=68" wheel packaging "cmake>=3.18" ninja "pybind11>=2.6"

# TE 2.8 is pinned because its public installation and FP8 recipe APIs are the
# contract implemented by this repository. Build its PyTorch extension against
# the already-installed private Torch wheel instead of resolving another Torch.
export NVTE_FRAMEWORK=pytorch
export MAX_JOBS
export NVTE_BUILD_THREADS_PER_JOB=1
export NVTE_CUDA_INCLUDE_PATH="$CUDA_HOME/include"
export CUDNN_HOME
export CUDNN_PATH="$CUDNN_HOME"
export NCCL_HOME
export NCCL_PATH="$NCCL_HOME"
export CMAKE_PREFIX_PATH="$CUDNN_HOME:$NCCL_HOME${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CPATH="$CUDNN_HOME/include:$NCCL_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDNN_HOME/lib:$NCCL_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:$NCCL_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv pip install --python "$FP8_ENV/bin/python" --no-build-isolation \
  "transformer-engine[pytorch]==${TE_VERSION}"

# Match the text workload stack already validated for the project and vLLM.
uv pip install --python "$FP8_ENV/bin/python" \
  "transformers==4.57.6" \
  "tokenizers==0.22.1" \
  "huggingface-hub>=0.34,<1" \
  "datasets>=2.20,<5"

"$FP8_ENV/bin/python" "$PROJECT_ROOT/scripts/check_private_torch.py" \
  --expected-environment "$FP8_ENV" \
  --expected-version 2.8.0 \
  --expected-cuda 12.4 \
  --require-gloo \
  --require-nccl \
  --require-cuda

export HPC_AI_FP8_ENV="$FP8_ENV"
export HPC_AI_TE_VERSION="$TE_VERSION"
"$FP8_ENV/bin/python" - <<'PY'
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import torch
import transformer_engine.pytorch as te

environment = Path(os.environ["HPC_AI_FP8_ENV"])
observed_te = version("transformer-engine")
expected_te = os.environ["HPC_AI_TE_VERSION"]
if observed_te != expected_te:
    raise SystemExit(f"Transformer Engine version mismatch: expected {expected_te}, observed {observed_te}")
if not hasattr(te, "Linear"):
    raise SystemExit("Transformer Engine PyTorch bindings do not expose Linear")
manifest = {
    "schema_version": "1.0",
    "python_version": platform.python_version(),
    "torch_version": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
    "transformer_engine_version": observed_te,
    "transformer_engine_linear": True,
    "cudnn_home": os.environ["CUDNN_HOME"],
    "cudnn_runtime": torch.backends.cudnn.version(),
    "nccl_home": os.environ["NCCL_HOME"],
    "nccl_runtime": torch.cuda.nccl.version(),
    "environment": str(environment),
}
(environment / "BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PY

echo "Transformer Engine FP8 environment is ready: $FP8_ENV"
