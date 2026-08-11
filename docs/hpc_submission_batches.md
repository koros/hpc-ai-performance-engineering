# HPC Submission Batches

This page defines the first submission order after the smoke jobs pass. Generated scripts are written under `slurm/generated/`, which is intentionally ignored by Git.

## Prerequisites

Complete the smoke runbook first:

```bash
docs/hpc_smoke_runbook.md
```

Then refresh derived reports:

```bash
python -m hpc_ai_perf.cli refresh
python -m hpc_ai_perf.cli readiness-report
```

## Cluster Profile

Create a local Slurm profile from the tracked example:

```bash
cp slurm/profiles/example.json slurm/profiles/cluster.local.json
```

Edit the local profile for the target site:

- `partition`
- `account`, only if required by the cluster
- `setup_commands`
- `modules`, only if not using a project-local setup script
- `environment`

The default local profile assumes jobs are submitted from the project root. Load the site-required CUDA, Python, and PyTorch modules before activating `.venv`; on Lovelace the profile uses:

```bash
source slurm/modules_for_hpc_ai
source .venv/bin/activate
```

Review `slurm/modules_for_hpc_ai` on the cluster before submitting. If the module stack changes, edit that helper or replace it in `setup_commands` with the appropriate `module load ...` commands.

Use the profile while rendering:

```bash
SLURM_PROFILE=slurm/profiles/cluster.local.json make check-slurm-profile
SLURM_PROFILE=slurm/profiles/cluster.local.json make render-slurm-first-batches
```

Generate reviewable submission manifests:

```bash
SLURM_PROFILE=slurm/profiles/cluster.local.json make plan-slurm-first-batches
```

Initialize job-status tracking before submission:

```bash
make init-slurm-first-status
make dry-run-submit-slurm-first
```

## Batch 1: Environment Validation

Purpose: confirm CUDA, PyTorch DDP/NCCL, FSDP, and inference serving dependencies on the target cluster.

Generate scripts:

```bash
make render-slurm-env-validation
```

Or with site defaults:

```bash
SLURM_PROFILE=slurm/profiles/cluster.local.json make render-slurm-env-validation
```

Expected scripts:

| Experiment | Purpose |
| --- | --- |
| `E001` | CUDA/PyTorch 1-GPU validation |
| `E002` | DDP/NCCL 2-GPU validation |
| `E003` | DDP/NCCL 4-GPU validation |
| `E004` | FSDP 2-GPU validation |
| `E006` | vLLM 1-GPU inference validation |

Review before submission:

```bash
ls slurm/generated/environment_validation_must_have/*.sbatch
sed -n '1,160p' slurm/generated/environment_validation_must_have/SUBMISSION_PLAN.md
sed -n '1,20p' results/processed/slurm_environment_validation_status.csv
```

## Batch 2: Baseline Profiling

Purpose: establish single-GPU baseline training and inference behavior for GPT-2 Small, DistilBERT, and TinyLlama inference.

Generate scripts:

```bash
make render-slurm-baseline-must-have
```

Or with site defaults:

```bash
SLURM_PROFILE=slurm/profiles/cluster.local.json make render-slurm-baseline-must-have
```

Expected scripts:

| Experiment range | Workload |
| --- | --- |
| `E008-E011` | GPT-2 Small training and inference baselines |
| `E012-E015` | DistilBERT training and inference baselines |
| `E018-E019` | TinyLlama inference baselines |

Review before submission:

```bash
ls slurm/generated/baseline_profiling_must_have/*.sbatch
sed -n '1,160p' slurm/generated/baseline_profiling_must_have/SUBMISSION_PLAN.md
sed -n '1,20p' results/processed/slurm_baseline_profiling_status.csv
```

## Batch 1 + 2 Convenience Target

Generate both batches:

```bash
make render-slurm-first-batches
```

Generate scripts and submission manifests:

```bash
make plan-slurm-first-batches
```

Generate status CSVs:

```bash
make init-slurm-first-status
```

## Recording Job IDs

Dry-run the first submission commands:

```bash
make dry-run-submit-slurm-first
```

On the cluster, submit a generated batch and automatically record returned job IDs:

```bash
python -m hpc_ai_perf.cli submit-slurm-batch \
  --script-dir slurm/generated/environment_validation_must_have \
  --status results/processed/slurm_environment_validation_status.csv
```

After a manual `sbatch` command, record the returned job ID against the experiment:

```bash
python -m hpc_ai_perf.cli record-slurm-job \
  --status results/processed/slurm_environment_validation_status.csv \
  --experiment-id E001 \
  --sbatch-output "Submitted batch job 123456"
```

Generate Markdown status reports:

```bash
make report-slurm-first-status
```

## After Each Batch

Refresh result-derived outputs:

```bash
python -m hpc_ai_perf.cli refresh
```

Then review:

- `results/processed/experiment_tracker.md`
- `results/processed/readiness_report.md`
- dashboard at `http://localhost:8000/dashboard/`
