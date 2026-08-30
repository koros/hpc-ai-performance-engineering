# Lovelace vLLM Environment Runbook

## Purpose

This runbook creates a private, reproducible vLLM environment without changing the repository `.venv`. The shared
`.venv` remains the certified runtime for all non-vLLM PyTorch and DeepSpeed waves. vLLM is source-built because it
must match Lovelace's Torch 2.8.0, CUDA 12.4.1, driver, and compiler stack.

The process has two gates:

1. Build pinned vLLM `0.10.2` in an external environment.
2. Run E006 as a one-GPU acceptance job and require import, engine start, and one served request.

After both gates pass, submit the already-defined canonical L40S E006 pilot. The acceptance JSON is not matrix evidence
and is never imported into `results/raw/`; the other vLLM rows remain deferred until the canonical pilot is reviewed.

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

## Build on L40S

From the project root on the login node, submit the build:

```bash
make submit-vllm-build
squeue -u "$USER"
```

The job runs [`slurm/setup_vllm_environment.sh`](../slurm/setup_vllm_environment.sh), which does all of the
following:

- loads the verified Spack 217 Lovelace module stack;
- checks `nvcc`, Clang, Torch 2.8.x, and CUDA 12.4;
- creates `~/.local/share/hpc-ai-performance-engineering/vllm-0.10.2` with `--system-site-packages`;
- clones the exact `v0.10.2` source tag outside the repository;
- bootstraps private `setuptools` 77–80 and vLLM `0.10.2`'s undeclared editable-build metadata dependency
  (`setuptools_scm`) before building with `uv pip install --no-build-isolation -e .` against the module Torch runtime;
  and
- targets both L40S (`sm_89`) and H100 (`sm_90`) by default.

The Slurm log paths are printed by `sbatch` and are named `slurm/vllm-build-<job-id>.out` and `.err`. Do not remove a
failed external environment or source checkout automatically. Read the log first; the setup script refuses to overwrite
an unexpected existing directory or source revision. Use a new `HPC_AI_VLLM_ENV` / `HPC_AI_VLLM_SOURCE` path if a clean
retry is required.

## Run the E006 Acceptance Gate

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
`HPC_AI_VLLM_ENV=/absolute/path/to/vllm-0.10.2` in that `--export` list as well.

## After Acceptance

Do not manually edit or import the acceptance record. After reviewing its successful status and JSON, submit the
dedicated immutable L40S E006 pilot:

```bash
source slurm/modules_for_hpc_ai
source .venv/bin/activate
make submit-vllm-pilot
```

Monitor and collect it through the standard lifecycle:

```bash
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-pilot
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-pilot
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-pilot
git commit -m "data: add completed L40S vLLM E006 trial"
```

`submit-vllm-pilot` freezes the platform-resolved E006 configuration and activates the isolated vLLM environment inside
the bundle. Once the canonical pilot produces valid evidence, review it before re-enabling the other 36 vLLM rows in
controlled waves. `VENV_ACTIVATE` remains available for those later vLLM waves and should not be set for non-vLLM work.
