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

## Batch 1: Environment Validation

Purpose: confirm CUDA, PyTorch DDP/NCCL, FSDP, and inference serving dependencies on the target cluster.

Generate scripts:

```bash
make render-slurm-env-validation
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
```

## Batch 2: Baseline Profiling

Purpose: establish single-GPU baseline training and inference behavior for GPT-2 Small, DistilBERT, and TinyLlama inference.

Generate scripts:

```bash
make render-slurm-baseline-must-have
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
```

## Batch 1 + 2 Convenience Target

Generate both batches:

```bash
make render-slurm-first-batches
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
