# Lovelace vLLM Environment Runbook

## Purpose

This runbook creates a private, reproducible PyTorch and vLLM stack without changing the repository `.venv`. The
shared `.venv` remains the certified runtime for all non-vLLM PyTorch and DeepSpeed waves. PyTorch and vLLM are both
source-built because the site PyTorch 2.8.0/CUDA 12.4 module omits Gloo, while vLLM uses Gloo for CPU-side process
coordination in addition to NCCL for GPU communication.

The process has three gates:

1. Build a reusable private PyTorch `2.8.0`/CUDA `12.4` wheel with CUDA, NCCL, and Gloo enabled.
2. Build pinned vLLM `0.10.2` against that exact private wheel.
3. Run E006 as a one-GPU acceptance job and require import, engine start, and one served request.

All three gates have passed on L40S. The acceptance JSON is diagnostic only and is never imported into `results/raw/`;
E006 must still run once through the immutable `vllm-pilot` wave before the performance waves are treated as released.
H100 vLLM conditions remain deferred until this same acceptance gate passes on H100.

Canonical `vllm-*` submission bundles treat this private environment as isolated: after loading the cluster modules,
the generated Slurm script clears inherited `PYTHONPATH`/`PYTHONHOME`, activates the private environment, and then
sets `PYTHONPATH` only to the immutable bundle snapshot. This prevents the module-provided Torch build from shadowing
the accepted private Gloo/NCCL-enabled build. Ordinary project `.venv` waves retain the module Python path because
they intentionally consume the cluster's site packages.

## One-Time Preparation

Install `uv` in your user account if the command is unavailable:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new login shell, or make its standard install location visible:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

Ensure the repository source is committed and clean. The build itself is external, so it will not make the repository
or later submission snapshots dirty.

The PyTorch source checkout, build environment, and wheel can require substantial home-directory space. Check both free
space and your user quota before submitting:

```bash
df -h "$HOME/.local/share"
quota -s 2>/dev/null || true
```

Do not activate `.venv` or the old `vllm-0.10.2` environment before these Make targets. Each Slurm job establishes its
own module and Python environment.

## Gate 1: Build Gloo-Enabled PyTorch

From the project root on the login node:

```bash
make submit-pytorch-gloo-build
squeue -u "$USER"
```

This submits [`slurm/smoke/pytorch_gloo_build_l40s.sbatch`](../slurm/smoke/pytorch_gloo_build_l40s.sbatch). The build
uses one L40S so it can validate the finished wheel against a real CUDA device before publishing its manifest. It:

- loads the verified Spack 217 compiler and CUDA 12.4.1 paths;
- removes Spack's Python package paths before creating the private environment;
- checks out the exact PyTorch `v2.8.0` source and its submodules outside the repository;
- builds for both L40S (`sm_89`) and H100 (`sm_90`) with `USE_CUDA=1`, `USE_DISTRIBUTED=1`, `USE_GLOO=1`, and
  `USE_NCCL=1`;
- disables OpenMP, FBGEMM, and oneDNN/MKLDNN because the site LLVM `libomp` has unresolved hwloc symbols and these
  CPU acceleration paths are not required by the CUDA-only vLLM profile or Gloo transport;
- preserves a build directory configured with the incompatible OpenMP profile under a timestamped
  `build-openmp-incompatible-*` name before starting the corrected configuration;
- installs the wheel into a private validation environment and completes one-rank NCCL/GPU and Gloo/CPU all-reduces;
- writes `BUILD_MANIFEST.json` containing the source commit, build options, wheel name, and SHA-256.

Default external paths are:

```text
~/.local/share/hpc-ai-performance-engineering/pytorch-2.8.0-cu124-gloo-build
~/.local/share/hpc-ai-performance-engineering/src/pytorch-v2.8.0
~/.local/share/hpc-ai-performance-engineering/wheels/pytorch-2.8.0-cu124-gloo
```

Inspect completion and logs using the job ID printed by `sbatch`:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
tail -n 120 slurm/pytorch-gloo-build-JOB_ID.out
tail -n 200 slurm/pytorch-gloo-build-JOB_ID.err
```

Do not proceed unless the job is `COMPLETED` and its output reports passing private import, CUDA, Gloo, and NCCL checks.
The target is safe to resubmit after success: it verifies and reuses the existing wheel instead of recompiling it.

## Gate 2: Build vLLM on L40S

After Gate 1 passes, submit the vLLM build:

```bash
make submit-vllm-build
squeue -u "$USER"
```

The job runs [`slurm/setup_vllm_environment.sh`](../slurm/setup_vllm_environment.sh), which does all of the
following:

- loads the verified Spack 217 Lovelace module stack;
- checks `nvcc`, Clang, and CUDA 12.4;
- verifies the private PyTorch wheel against its SHA-256 build manifest;
- creates `~/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo` without `--system-site-packages`;
- installs and validates the private PyTorch wheel before compiling vLLM;
- clones the exact `v0.10.2` source tag outside the repository;
- bootstraps private `setuptools` 77–80 and vLLM `0.10.2`'s undeclared editable-build metadata dependency
  (`setuptools_scm`) before building with verbose `uv pip install --no-build-isolation -e .` output against the
  private Torch runtime; and
- passes `-fbracket-depth=1024` to the selected Clang host compiler through `nvcc`, which is required for vLLM's
  generated Marlin CUDA source on this stack; and
- targets both L40S (`sm_89`) and H100 (`sm_90`) by default.

The installed source build reports `vllm==0.10.2+cu124`: the `+cu124` suffix records the linked CUDA toolkit. The
build gate verifies the pinned base version (`0.10.2`) and the private Torch/CUDA runtime, so this expected local version
suffix is accepted.

vLLM `0.10.2` allows Transformers `>=4.55.2`, but the current release line can otherwise resolve Transformers 5,
which removed a GPT-2 tokenizer API used by these workloads. The isolated profile pins `transformers==4.57.6` after
building vLLM, alongside its compatible `tokenizers==0.22.1` and the matching Transformers Python dependencies (such
as `huggingface-hub`). These packages do not require Torch, so this does not replace the private Torch/CUDA runtime.

The current vLLM matrix consists only of text workloads (GPT-2, TinyLlama, and DistilBERT). The build therefore removes
PyPI TorchAudio and TorchVision from the isolated environment by default: their native wheels are not built against this
private CUDA 12.4 Torch. This permits Transformers and vLLM's text `LLM` entry point to import normally.
Do not add audio or multimodal-audio workloads to this profile. First provide a TorchAudio build compiled for the same
Torch/CUDA runtime, then set `HPC_AI_VLLM_TEXT_ONLY=0` when building.

The Slurm log paths are printed by `sbatch` and are named `slurm/vllm-build-<job-id>.out` and `.err`. Verbose mode
retains CMake/Ninja compiler diagnostics in the `.err` log if a native extension fails. Do not remove a
failed external environment or source checkout automatically. Read the log first; the setup script refuses to overwrite
an unexpected existing directory or source revision. Use a new `HPC_AI_VLLM_ENV` / `HPC_AI_VLLM_SOURCE` path if a clean
retry is required.

## Gate 3: Run the E006 Acceptance Gate

After the build job completes successfully:

```bash
make submit-vllm-acceptance
squeue -u "$USER"
```

The gate runs on one L40S by default and performs:

```text
check-runtime --require-vllm
E006: vLLM import → engine start → pinned GPT-2 request served
```

vLLM 0.10.2 creates a Gloo CPU process group even for its one-GPU engine. Before running E006, the acceptance script
verifies that `torch.__file__` is inside the private environment and requires CUDA, Gloo, and NCCL. This prevents the
site PyTorch module from silently re-entering the environment through `PYTHONPATH`; it also repeats the one-rank NCCL
and Gloo all-reduce smoke before starting vLLM.

The adapters pass pre-tokenized requests through vLLM 0.10.2's positional `prompts` argument using the `TokensPrompt`
schema (`{"prompt_token_ids": [...]}`). The older `LLM.generate(prompt_token_ids=...)` keyword form is not accepted by
the pinned API. After writing the result, the acceptance job validates the current Slurm job's JSON and exits non-zero
unless E006 completed with a served request and positive generated-token throughput.

Its result is retained under `build/vllm_acceptance/l40s/`; review the JSON and job log. Success requires:

- `status: completed`;
- `vllm_import: true`;
- `engine_start: true`; and
- `request_served: true`.

For an H100 confirmation after the L40S gate passes, submit the same script with the scheduler partition and expected
model overridden:

```bash
sbatch --partition=gpu_h100 \
  --export=ALL,HPC_AI_EXPECTED_GPU_MODEL=H100 \
  slurm/smoke/vllm_acceptance.sbatch
```

The H100 job uses the same multi-architecture private build. If the user chose a custom environment location, include
`HPC_AI_VLLM_ENV=/absolute/path/to/vllm-0.10.2-gloo` in that `--export` list as well.

## Run Canonical L40S vLLM Waves

Do not manually import the diagnostic acceptance record. Submit E006 as canonical matrix evidence first:

```bash
make submit-vllm-pilot
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-pilot
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-pilot
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-pilot
```

After the pilot reports one provenance-valid completed trial, run the performance waves in this order:

| Wave | Conditions | Trials | Purpose |
| --- | --- | ---: | --- |
| `vllm-framework` | `E130`, `E178` | 3 | GPT-2 and TinyLlama framework baselines. |
| `vllm-batching` | `E132-E137`, `E180-E185` | 3 | Serving batch-size sweeps. |
| `vllm-concurrency` | `E138-E142`, `E186-E190` | 3 | Concurrent-request sweeps. |
| `vllm-prompt` | `E143-E146`, `E191-E194` | 3 | Prompt-length sweeps. |
| `vllm-tp2` | `E147`, `E195` | 3 | Two-GPU tensor-parallel serving. |

For example:

```bash
make submit-vllm-wave PLATFORM=l40s PLATFORM_WAVE=vllm-framework
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-framework
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-framework
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-framework
```

`submit-vllm-wave` always supplies the private environment to the immutable bundle. Never supply that environment to
non-vLLM waves. L40S `E148` and `E196` remain deferred because they request four GPUs; the active node exposes two.
