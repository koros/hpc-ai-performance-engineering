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

Edit the partition if needed, replace the module-loading helper if the site stack changes, and add an account line only if the target HPC site requires one.

After smoke jobs complete:

```bash
python -m hpc_ai_perf.cli validate
python -m hpc_ai_perf.cli refresh
```
