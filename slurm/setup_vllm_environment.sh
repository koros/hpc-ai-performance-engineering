#!/usr/bin/env bash
# Build vLLM in an isolated virtual environment for the verified Lovelace stack.
#
# This intentionally leaves the repository's .venv unchanged: that environment
# remains the known-good runtime for the non-vLLM experiment waves. The vLLM
# source tree and virtual environment live outside the repository so generated
# build files cannot make a submission snapshot dirty.
set -euo pipefail

VLLM_VERSION="${HPC_AI_VLLM_VERSION:-0.10.2}"
VLLM_TAG="v${VLLM_VERSION}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_BASE="${HPC_AI_VLLM_BASE:-${HOME}/.local/share/hpc-ai-performance-engineering}"
VLLM_ENV="${HPC_AI_VLLM_ENV:-${VLLM_BASE}/vllm-${VLLM_VERSION}}"
VLLM_SOURCE="${HPC_AI_VLLM_SOURCE:-${VLLM_BASE}/src/vllm-${VLLM_TAG}}"
VLLM_MAX_JOBS="${HPC_AI_VLLM_MAX_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
VLLM_CLANG_BRACKET_DEPTH="${HPC_AI_VLLM_CLANG_BRACKET_DEPTH:-1024}"
VLLM_TEXT_ONLY="${HPC_AI_VLLM_TEXT_ONLY:-1}"
VLLM_TRANSFORMERS_VERSION="${HPC_AI_VLLM_TRANSFORMERS_VERSION:-4.57.6}"
VLLM_TOKENIZERS_VERSION="${HPC_AI_VLLM_TOKENIZERS_VERSION:-0.22.1}"

usage() {
  cat <<'EOF'
Usage: bash slurm/setup_vllm_environment.sh [options]

Build the pinned vLLM source release into a separate virtual environment.

Options:
  --project-root PATH  Repository containing slurm/modules_for_hpc_ai.
  --env-dir PATH       Destination virtual environment.
  --source-dir PATH    Destination vLLM source checkout.
  --max-jobs COUNT     Parallel compiler jobs (default: SLURM_CPUS_PER_TASK or 4).
  --help               Show this message.

Environment overrides: HPC_AI_VLLM_VERSION, HPC_AI_VLLM_BASE,
HPC_AI_VLLM_ENV, HPC_AI_VLLM_SOURCE, HPC_AI_VLLM_MAX_JOBS, and
HPC_AI_VLLM_TORCH_CUDA_ARCH_LIST. HPC_AI_VLLM_CLANG_BRACKET_DEPTH defaults
to 1024 for vLLM's generated Marlin code. The default architecture list is
8.9;9.0, so one isolated build can be used on the L40S and H100 nodes.
HPC_AI_VLLM_TEXT_ONLY defaults to 1: it removes an incompatible PyPI
TorchAudio wheel because the current experiment matrix contains text-only
serving workloads. Set it to 0 only after providing TorchAudio built for the
cluster's Torch/CUDA runtime.
HPC_AI_VLLM_TRANSFORMERS_VERSION defaults to 4.57.6, the pinned Transformers
release compatible with vLLM 0.10.2's text-model tokenizer API.
HPC_AI_VLLM_TOKENIZERS_VERSION defaults to 0.22.1, a release accepted by the
pinned Transformers runtime.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --env-dir)
      VLLM_ENV="$2"
      shift 2
      ;;
    --source-dir)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    --max-jobs)
      VLLM_MAX_JOBS="$2"
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

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
VLLM_ENV="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$VLLM_ENV")"
VLLM_SOURCE="$(python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$VLLM_SOURCE")"

if [[ ! "$VLLM_MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --max-jobs must be a positive integer; got $VLLM_MAX_JOBS" >&2
  exit 2
fi
if [[ ! "$VLLM_CLANG_BRACKET_DEPTH" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: HPC_AI_VLLM_CLANG_BRACKET_DEPTH must be a positive integer; got $VLLM_CLANG_BRACKET_DEPTH" >&2
  exit 2
fi
if [[ "$VLLM_TEXT_ONLY" != "0" && "$VLLM_TEXT_ONLY" != "1" ]]; then
  echo "ERROR: HPC_AI_VLLM_TEXT_ONLY must be 0 or 1; got $VLLM_TEXT_ONLY" >&2
  exit 2
fi

# A normal venv activation only prepends one path entry. Strip it in this
# child process so the virtual environment is created from the module-provided
# Python, even if the caller happens to have the project .venv activated.
if [[ -n "${VIRTUAL_ENV:-}" && "$PATH" == "$VIRTUAL_ENV/bin:"* ]]; then
  export PATH="${PATH#"$VIRTUAL_ENV/bin:"}"
fi
unset VIRTUAL_ENV PYTHONHOME
hash -r

if [[ ! -f "$PROJECT_ROOT/slurm/modules_for_hpc_ai" ]]; then
  echo "ERROR: project root does not contain slurm/modules_for_hpc_ai: $PROJECT_ROOT" >&2
  exit 2
fi

# The project helper is the single source of truth for the site module stack,
# nvcc, CUDA_HOME, and the compiler selected by Laurence's validated modules.
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/slurm/modules_for_hpc_ai"

if ! command -v uv >/dev/null 2>&1 && [[ -x "${HOME}/.local/bin/uv" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: uv is not available on PATH.
Install it once in your user account, then rerun this job:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
  exit 3
fi

for command in nvcc clang clang++ git; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required build command is unavailable: $command" >&2
    exit 3
  fi
done

BOOTSTRAP_PYTHON="$(command -v python3)"
"$BOOTSTRAP_PYTHON" - <<'PY'
import torch

if not torch.__version__.startswith("2.8.") or torch.version.cuda != "12.4":
    raise SystemExit(
        f"Expected the Lovelace Torch 2.8/CUDA 12.4 stack; found torch {torch.__version__}, CUDA {torch.version.cuda}."
    )
print(f"Using module Torch {torch.__version__} with CUDA {torch.version.cuda}")
PY

if [[ -e "$VLLM_ENV" && ! -x "$VLLM_ENV/bin/python" ]]; then
  echo "ERROR: vLLM environment path exists but is not a virtual environment: $VLLM_ENV" >&2
  exit 3
fi
if [[ ! -e "$VLLM_ENV" ]]; then
  mkdir -p "$(dirname "$VLLM_ENV")"
  "$BOOTSTRAP_PYTHON" -m venv --system-site-packages "$VLLM_ENV"
fi

if [[ -e "$VLLM_SOURCE" ]]; then
  if [[ ! -d "$VLLM_SOURCE/.git" ]]; then
    echo "ERROR: vLLM source path exists but is not a git checkout: $VLLM_SOURCE" >&2
    exit 3
  fi
  existing_tag="$(git -C "$VLLM_SOURCE" describe --tags --exact-match HEAD 2>/dev/null || true)"
  if [[ "$existing_tag" != "$VLLM_TAG" ]]; then
    echo "ERROR: refusing to modify existing vLLM checkout at $VLLM_SOURCE (expected $VLLM_TAG, found ${existing_tag:-untagged})." >&2
    echo "       Choose a new HPC_AI_VLLM_SOURCE path or inspect that checkout first." >&2
    exit 3
  fi
else
  mkdir -p "$(dirname "$VLLM_SOURCE")"
  git clone --branch "$VLLM_TAG" --depth 1 https://github.com/vllm-project/vllm.git "$VLLM_SOURCE"
fi

export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export MAX_JOBS="$VLLM_MAX_JOBS"
export TORCH_CUDA_ARCH_LIST="${HPC_AI_VLLM_TORCH_CUDA_ARCH_LIST:-8.9;9.0}"

# CUDA 12.4 forwards the generated Marlin source to the selected Clang host
# compiler. vLLM 0.10.2's gptq_marlin.cu exceeds Clang 14's default nesting
# limit of 256, so initialise CMake's CUDA flags with the documented host
# compiler override. CUDAFLAGS becomes CMAKE_CUDA_FLAGS during configure.
if "$CXX" --version 2>&1 | grep -qi 'clang version'; then
  export CUDAFLAGS="${CUDAFLAGS:+${CUDAFLAGS} }-Xcompiler=-fbracket-depth=${VLLM_CLANG_BRACKET_DEPTH}"
fi

# The Spack modules expose their Python packages through PYTHONPATH. That path
# otherwise wins over a --system-site-packages venv and makes the build import
# the site's setuptools 76.1 instead of the private compatible version below.
# Keep the module paths for Torch, but put this venv's site-packages first.
VLLM_SITE_PACKAGES="$("$VLLM_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="${VLLM_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Building $VLLM_TAG"
echo "  environment: $VLLM_ENV"
echo "  source:      $VLLM_SOURCE"
echo "  compiler:    $CC / $CXX"
echo "  CUDA_HOME:   $CUDA_HOME"
echo "  architectures: $TORCH_CUDA_ARCH_LIST"
echo "  CUDAFLAGS:   ${CUDAFLAGS:-unset}"

# vLLM documents uv pip install -e . for a source build. The target environment
# inherits the Spack Torch runtime read-only, while its vLLM/Python packages are
# installed privately. No package is written into the project .venv.
cd "$VLLM_SOURCE"
# vLLM v0.10.2 imports setuptools_scm while preparing editable metadata but
# does not declare it in its build-system requirements. Its SPDX license value
# also needs setuptools 77 or newer, while the site Python provides 76.1.
# Build isolation is deliberately disabled so the extension sees the
# module-provided Torch ABI; bootstrap these metadata requirements privately.
uv pip install --python "$VLLM_ENV/bin/python" "setuptools>=77,<81" "setuptools_scm[toml]>=8,<9"
"$VLLM_ENV/bin/python" - <<'PY'
import setuptools

major = int(setuptools.__version__.split(".", 1)[0])
if major < 77:
    raise SystemExit(f"Expected private setuptools >=77; found {setuptools.__version__} at {setuptools.__file__}")
print(f"Using private setuptools {setuptools.__version__} at {setuptools.__file__}")
PY
# Native extension errors are emitted by CMake/Ninja under the package builder.
# Keep uv verbose so the Slurm stderr log contains the compiler diagnostic rather
# than only the final editable-wheel traceback. This makes a failed build
# actionable without having to recover an ephemeral uv build directory.
uv --verbose pip install --python "$VLLM_ENV/bin/python" --no-build-isolation -e .

# vLLM 0.10.2 permits any Transformers release >=4.55.2, but Transformers 5
# removed GPT-2 tokenizer API used by the current text-serving workloads.
# Resolve the complete Transformers 4.x Python dependency set after the vLLM
# resolver has completed. In particular, this replaces a Transformers 5-era
# huggingface-hub/tokenizers combination with one accepted by 4.57.6. Neither
# package requires Torch, so this leaves the module-provided CUDA runtime
# untouched.
echo "Resolving Transformers $VLLM_TRANSFORMERS_VERSION and Tokenizers $VLLM_TOKENIZERS_VERSION for vLLM text serving"
uv pip install --python "$VLLM_ENV/bin/python" \
  "transformers==${VLLM_TRANSFORMERS_VERSION}" \
  "tokenizers==${VLLM_TOKENIZERS_VERSION}"

# PyPI's TorchAudio 2.8 wheel is built against CUDA 12.8, but the cluster
# provides Torch 2.8 compiled for CUDA 12.4. Transformers imports TorchAudio
# when it is present, even for a text LLM, which prevents vLLM's LLM entry
# point from loading. The current matrix contains only text workloads, so
# remove this optional incompatible wheel. A future audio workload must use a
# TorchAudio build that matches the cluster runtime and set TEXT_ONLY=0.
if [[ "$VLLM_TEXT_ONLY" == "1" ]]; then
  echo "Removing incompatible optional TorchAudio wheel for the text-only vLLM profile"
  "$VLLM_ENV/bin/python" -m pip uninstall -y torchaudio
fi

"$VLLM_ENV/bin/python" - <<PY
from importlib.metadata import version
from packaging.version import Version
import torch
from transformers import ProcessorMixin
from vllm import LLM, SamplingParams

installed_vllm_version = version("vllm")
installed_transformers_version = version("transformers")
installed_tokenizers_version = version("tokenizers")
print("vllm", installed_vllm_version)
print("transformers", installed_transformers_version)
print("tokenizers", installed_tokenizers_version)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
if Version(installed_vllm_version).base_version != "$VLLM_VERSION":
    raise SystemExit("vLLM base version does not match the pinned source tag")
if not torch.__version__.startswith("2.8.") or torch.version.cuda != "12.4":
    raise SystemExit("isolated environment no longer resolves the verified Torch 2.8/CUDA 12.4 runtime")
if installed_transformers_version != "$VLLM_TRANSFORMERS_VERSION":
    raise SystemExit("isolated environment no longer resolves the pinned Transformers runtime")
if installed_tokenizers_version != "$VLLM_TOKENIZERS_VERSION":
    raise SystemExit("isolated environment no longer resolves the pinned Tokenizers runtime")
print("transformers ProcessorMixin import OK")
print("vLLM LLM and SamplingParams imports OK")
PY

cat <<EOF
vLLM build completed successfully.

Use this exact environment for the serving acceptance job:
  export HPC_AI_VLLM_ENV=$VLLM_ENV
  sbatch $PROJECT_ROOT/slurm/smoke/vllm_acceptance.sbatch

Do not un-defer E006 or the wider vLLM rows until that job completes with
vllm_import, engine_start, and request_served all true.
EOF
