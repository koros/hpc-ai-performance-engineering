# Experiment Configs

This directory contains runnable experiment configs and reusable templates.

| Path | Purpose |
| --- | --- |
| `templates/` | Valid reusable JSON configs for each experiment family |
| `environment_validation/` | Concrete `E001-E007` startup validation configs |
| `baseline_profiling/` | Concrete `E008-E019` single-GPU training and inference baselines |
| `local/` | Local smoke-test configs that do not require HPC GPUs |
| `training/` | Concrete training configs |
| `inference/` | Concrete inference configs |
| `distributed/` | Distributed strategy configs and generated variants |
| `profiling/` | Profiling-focused configs and generated variants |

Concrete experiment configs should preserve the matrix ID in both `experiment_id` and `metadata.matrix_id`.
