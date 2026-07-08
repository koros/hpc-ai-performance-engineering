# Slurm Smoke Presets

These scripts run the optional transformer smoke configs on an HPC GPU node. They are intended for quick dependency and scheduler validation before submitting the full `E001-E200` experiment matrix.

Before submitting, prepare the environment on the cluster:

```bash
python -m pip install -e ".[workloads]"
```

Submit one smoke job:

```bash
sbatch slurm/smoke/gpt2_small_smoke.sbatch
sbatch slurm/smoke/distilbert_smoke.sbatch
sbatch slurm/smoke/tinyllama_inference_smoke.sbatch
```

Edit partition, account, module, and virtual environment lines for the target HPC site.
