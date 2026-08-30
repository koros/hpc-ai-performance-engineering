# Slurm Smoke Presets

These scripts run the optional transformer smoke configs on an HPC GPU node. They are intended for quick dependency and scheduler validation before submitting the full `E001-E200` experiment matrix.

See the full checklist in [`docs/hpc_smoke_runbook.md`](../../docs/hpc_smoke_runbook.md).

Before submitting, prepare the environment on the cluster:

```bash
cd ~/hpc-ai-performance-engineering
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,yaml,workloads]"
```

The scripts assume they are submitted from the project root. Load the required cluster modules for the target HPC site before activating `.venv`. On Lovelace, the scripts use `source slurm/modules_for_hpc_ai` as the project helper.

Submit one smoke job first, inspect its logs and raw JSON result, then submit the remaining presets:

```bash
sbatch slurm/smoke/gpt2_small_smoke.sbatch
sbatch slurm/smoke/distilbert_smoke.sbatch
sbatch slurm/smoke/tinyllama_inference_smoke.sbatch
```

## Isolated vLLM Acceptance

vLLM is not installed into the shared `.venv`: its build is coupled to the CUDA compiler and the module-provided
PyTorch ABI. The project provides a reproducible private build path pinned to vLLM `0.10.2`. It preserves `.venv` for
all supported PyTorch and DeepSpeed waves, and stores the vLLM environment outside the repository by default.

First install `uv` once in your user account if it is not already on `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project root, submit the build and wait for it to complete:

```bash
make submit-vllm-build
squeue -u "$USER"
```

The build uses the verified Lovelace module helper, Clang, CUDA 12.4.1, and an architecture list covering L40S
(`sm_89`) and H100 (`sm_90`). Its private paths default to
`~/.local/share/hpc-ai-performance-engineering/vllm-0.10.2` and can be changed with `HPC_AI_VLLM_ENV` and
`HPC_AI_VLLM_SOURCE`.

After the build log confirms success, run the single-GPU serving gate:

```bash
make submit-vllm-acceptance
```

The acceptance job runs `check-runtime --require-vllm` and E006's pinned GPT-2 request. It must record a successful
vLLM import, engine start, and served request. Its result is deliberately written to ignored `build/vllm_acceptance/`:
it is an environment gate, not canonical matrix evidence. After it passes, use `make submit-vllm-pilot` for the formal
one-trial E006 evidence; the remaining vLLM matrix rows stay deferred.

Edit the partition if needed, replace the module-loading helper if the site stack changes, and add an account line only if the target HPC site requires one.

After smoke jobs complete:

```bash
python -m hpc_ai_perf.cli validate
python -m hpc_ai_perf.cli refresh
```
