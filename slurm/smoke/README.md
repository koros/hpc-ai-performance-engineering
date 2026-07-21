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

The scripts assume they are submitted from the project root. They load cluster modules from
`slurm/modules_for_hpc_ai` and then activate `.venv`.

Submit one smoke job:

```bash
sbatch slurm/smoke/gpt2_small_smoke.sbatch
sbatch slurm/smoke/distilbert_smoke.sbatch
sbatch slurm/smoke/tinyllama_inference_smoke.sbatch
```

Edit the partition if needed, and add an account line only if the target HPC site requires one.
