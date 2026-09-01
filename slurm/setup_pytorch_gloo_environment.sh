#!/usr/bin/env bash
# Build a private PyTorch wheel with CUDA, NCCL, and Gloo enabled.
#
# The site py-torch module supplies the validated CUDA/compiler dependency
# stack, but its Python package is deliberately hidden while this script runs.
# The resulting wheel is installed into a normal virtual environment without
# --system-site-packages so vLLM cannot silently resolve the site Torch build.
set -euo pipefail

TORCH_VERSION="${HPC_AI_TORCH_VERSION:-2.8.0}"
TORCH_TAG="v${TORCH_VERSION}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIVATE_BASE="${HPC_AI_PRIVATE_BASE:-${HOME}/.local/share/hpc-ai-performance-engineering}"
TORCH_ENV="${HPC_AI_TORCH_ENV:-${PRIVATE_BASE}/pytorch-${TORCH_VERSION}-cu124-gloo-build}"
TORCH_SOURCE="${HPC_AI_TORCH_SOURCE:-${PRIVATE_BASE}/src/pytorch-${TORCH_TAG}}"
TORCH_WHEEL_DIR="${HPC_AI_TORCH_WHEEL_DIR:-${PRIVATE_BASE}/wheels/pytorch-${TORCH_VERSION}-cu124-gloo}"
TORCH_MAX_JOBS="${HPC_AI_TORCH_MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
TORCH_CUDA_ARCH_LIST="${HPC_AI_TORCH_CUDA_ARCH_LIST:-8.9;9.0}"

usage() {
  cat <<'EOF'
Usage: bash slurm/setup_pytorch_gloo_environment.sh [options]

Build PyTorch 2.8.0 from source against the validated CUDA 12.4 stack with
both Gloo and NCCL, then emit a reusable private wheel.

Options:
  --project-root PATH  Repository containing slurm/modules_for_hpc_ai.
  --env-dir PATH       Private build/validation virtual environment.
  --source-dir PATH    PyTorch source checkout.
  --wheel-dir PATH     Destination for the private wheel and build manifest.
  --max-jobs COUNT     Parallel compiler jobs.
  --help               Show this message.

Environment overrides: HPC_AI_TORCH_VERSION, HPC_AI_PRIVATE_BASE,
HPC_AI_TORCH_ENV, HPC_AI_TORCH_SOURCE, HPC_AI_TORCH_WHEEL_DIR,
HPC_AI_TORCH_MAX_JOBS, and HPC_AI_TORCH_CUDA_ARCH_LIST.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --env-dir)
      TORCH_ENV="$2"
      shift 2
      ;;
    --source-dir)
      TORCH_SOURCE="$2"
      shift 2
      ;;
    --wheel-dir)
      TORCH_WHEEL_DIR="$2"
      shift 2
      ;;
    --max-jobs)
      TORCH_MAX_JOBS="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$TORCH_MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max-jobs must be a positive integer; got $TORCH_MAX_JOBS" >&2
  exit 2
fi

# Do not let a caller's project venv select the bootstrap interpreter.
if [[ -n "${VIRTUAL_ENV:-}" && "$PATH" == "$VIRTUAL_ENV/bin:"* ]]; then
  export PATH="${PATH#"$VIRTUAL_ENV/bin:"}"
fi
unset VIRTUAL_ENV PYTHONHOME
hash -r

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
if [[ ! -f "$PROJECT_ROOT/slurm/modules_for_hpc_ai" ]]; then
  echo "ERROR: project root does not contain slurm/modules_for_hpc_ai: $PROJECT_ROOT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/slurm/modules_for_hpc_ai"

if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME}/.local/bin/uv" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
for command in uv nvcc clang clang++ git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required build command is unavailable: $command" >&2
    exit 3
  fi
done

BOOTSTRAP_PYTHON="$(command -v python3)"
TORCH_ENV="$($BOOTSTRAP_PYTHON -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$TORCH_ENV")"
TORCH_SOURCE="$($BOOTSTRAP_PYTHON -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$TORCH_SOURCE")"
TORCH_WHEEL_DIR="$($BOOTSTRAP_PYTHON -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$TORCH_WHEEL_DIR")"

# Spack publishes its Python packages through PYTHONPATH. Hiding that variable
# is the critical isolation boundary: compiler/CUDA library paths remain loaded,
# while the Gloo-disabled site Torch can no longer satisfy imports or builds.
MODULE_PYTHONPATH="${PYTHONPATH:-}"
unset PYTHONPATH PYTHONHOME

if [[ -e "$TORCH_ENV" && ! -x "$TORCH_ENV/bin/python" ]]; then
  echo "ERROR: Torch environment path exists but is not a virtual environment: $TORCH_ENV" >&2
  exit 3
fi
if [[ ! -e "$TORCH_ENV" ]]; then
  mkdir -p "$(dirname "$TORCH_ENV")"
  "$BOOTSTRAP_PYTHON" -m venv "$TORCH_ENV"
fi
export PATH="$TORCH_ENV/bin:$PATH"

if [[ -e "$TORCH_SOURCE" ]]; then
  if [[ ! -d "$TORCH_SOURCE/.git" ]]; then
    echo "ERROR: Torch source path exists but is not a git checkout: $TORCH_SOURCE" >&2
    exit 3
  fi
  existing_tag="$(git -C "$TORCH_SOURCE" describe --tags --exact-match HEAD 2>/dev/null || true)"
  if [[ "$existing_tag" != "$TORCH_TAG" ]]; then
    echo "ERROR: expected $TORCH_TAG at $TORCH_SOURCE; found ${existing_tag:-untagged}" >&2
    exit 3
  fi
  git -C "$TORCH_SOURCE" submodule update --init --recursive --depth 1
else
  mkdir -p "$(dirname "$TORCH_SOURCE")"
  git clone --branch "$TORCH_TAG" --depth 1 --recurse-submodules --shallow-submodules \
    https://github.com/pytorch/pytorch.git "$TORCH_SOURCE"
fi

mkdir -p "$TORCH_WHEEL_DIR"

# A successful wheel is reusable. Validate it rather than spending hours
# rebuilding when the job is safely resubmitted.
existing_wheels=("$TORCH_WHEEL_DIR"/torch-${TORCH_VERSION}-*.whl)
if [[ -e "${existing_wheels[0]}" ]]; then
  if [[ ${#existing_wheels[@]} -ne 1 ]]; then
    echo "ERROR: expected one private Torch wheel in $TORCH_WHEEL_DIR; found ${#existing_wheels[@]}" >&2
    exit 3
  fi
  TORCH_WHEEL="${existing_wheels[0]}"
  echo "Reusing existing private Torch wheel: $TORCH_WHEEL"
else
  # Install build requirements privately. The site Python package paths remain
  # hidden, preventing setuptools, NumPy, or Torch from leaking into the build.
  uv pip install --python "$TORCH_ENV/bin/python" -r "$TORCH_SOURCE/requirements.txt"
  uv pip install --python "$TORCH_ENV/bin/python" \
    "setuptools>=77,<81" wheel ninja "cmake>=3.27,<4"

  export CC="${CC:-clang}"
  export CXX="${CXX:-clang++}"
  export MAX_JOBS="$TORCH_MAX_JOBS"
  export TORCH_CUDA_ARCH_LIST
  export PYTORCH_BUILD_VERSION="$TORCH_VERSION"
  export PYTORCH_BUILD_NUMBER=1
  export USE_CUDA=1
  export USE_CUDNN=1
  export USE_DISTRIBUTED=1
  export USE_GLOO=1
  export USE_NCCL=1
  export USE_MPI=0
  # The site LLVM libomp has unresolved hwloc symbols. This private runtime is
  # exclusively for CUDA vLLM, so omit CPU OpenMP and the two CPU acceleration
  # libraries that probe it. Gloo itself uses its own thread/socket transport.
  export USE_OPENMP=0
  export USE_FBGEMM=0
  export USE_MKLDNN=0
  export BUILD_TEST=0

  # A previous attempt may have configured this checkout with the site OpenMP
  # runtime enabled. Reusing that CMake graph would keep the broken libomp/hwloc
  # probe alive even though the corrected options are now exported. Preserve
  # the partial build for diagnosis and force one clean reconfiguration.
  TORCH_BUILD_DIR="$TORCH_SOURCE/build"
  if [[ -f "$TORCH_BUILD_DIR/CMakeCache.txt" ]] && \
    grep -Eq '^USE_(OPENMP|FBGEMM|MKLDNN):BOOL=(ON|1)$' "$TORCH_BUILD_DIR/CMakeCache.txt"; then
    STALE_BUILD_DIR="${TORCH_SOURCE}/build-openmp-incompatible-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$TORCH_BUILD_DIR" "$STALE_BUILD_DIR"
    echo "Preserved incompatible OpenMP build cache at: $STALE_BUILD_DIR"
  fi

  echo "Building private PyTorch $TORCH_TAG"
  echo "  environment:   $TORCH_ENV"
  echo "  source:        $TORCH_SOURCE"
  echo "  wheel output:  $TORCH_WHEEL_DIR"
  echo "  compiler:      $CC / $CXX"
  echo "  CUDA_HOME:     $CUDA_HOME"
  echo "  architectures: $TORCH_CUDA_ARCH_LIST"
  echo "  max jobs:      $MAX_JOBS"

  cd "$TORCH_SOURCE"
  "$TORCH_ENV/bin/python" setup.py bdist_wheel --dist-dir "$TORCH_WHEEL_DIR"

  built_wheels=("$TORCH_WHEEL_DIR"/torch-${TORCH_VERSION}-*.whl)
  if [[ ${#built_wheels[@]} -ne 1 || ! -e "${built_wheels[0]}" ]]; then
    echo "ERROR: build did not produce exactly one Torch $TORCH_VERSION wheel" >&2
    exit 4
  fi
  TORCH_WHEEL="${built_wheels[0]}"
fi

uv pip install --python "$TORCH_ENV/bin/python" --reinstall "$TORCH_WHEEL"

# Leave the source directory before importing, otherwise Python can resolve the
# checkout instead of the installed wheel and give a false-positive validation.
cd "$PROJECT_ROOT"
"$TORCH_ENV/bin/python" "$PROJECT_ROOT/scripts/check_private_torch.py" \
  --expected-environment "$TORCH_ENV" \
  --expected-version "$TORCH_VERSION" \
  --expected-cuda 12.4 \
  --require-gloo \
  --require-nccl \
  --require-cuda \
  --smoke-process-groups

export HPC_AI_BUILT_TORCH_WHEEL="$TORCH_WHEEL"
export HPC_AI_BUILT_TORCH_SOURCE="$TORCH_SOURCE"
export HPC_AI_BUILT_TORCH_WHEEL_DIR="$TORCH_WHEEL_DIR"
export HPC_AI_BUILT_TORCH_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
export HPC_AI_BUILT_TORCH_COMMIT="$(git -C "$TORCH_SOURCE" rev-parse HEAD)"
"$TORCH_ENV/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

wheel = Path(os.environ["HPC_AI_BUILT_TORCH_WHEEL"])
manifest = {
    "torch_version": os.environ.get("HPC_AI_TORCH_VERSION", "2.8.0"),
    "torch_tag": "v" + os.environ.get("HPC_AI_TORCH_VERSION", "2.8.0"),
    "torch_source_commit": os.environ["HPC_AI_BUILT_TORCH_COMMIT"],
    "cuda_version": "12.4",
    "cuda_home": os.environ["CUDA_HOME"],
    "cuda_arch_list": os.environ["HPC_AI_BUILT_TORCH_ARCH_LIST"],
    "use_distributed": True,
    "use_gloo": True,
    "use_nccl": True,
    "use_openmp": False,
    "use_fbgemm": False,
    "use_mkldnn": False,
    "wheel": wheel.name,
    "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
}
(Path(os.environ["HPC_AI_BUILT_TORCH_WHEEL_DIR"]) / "BUILD_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

# Keep this diagnostic out of the manifest while making module leakage visible
# in the Slurm log when investigating a site stack change.
echo "Hidden module PYTHONPATH: ${MODULE_PYTHONPATH:-unset}"
cat <<EOF
Private PyTorch build completed successfully.

Wheel:
  $TORCH_WHEEL

Next:
  make submit-vllm-build
EOF
