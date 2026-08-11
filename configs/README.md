# Experiment Configs

This directory contains runnable experiment configs and reusable templates.

| Path | Purpose |
| --- | --- |
| `templates/` | Valid reusable JSON configs for each experiment family |
| `dev/` | Local Mac development baselines and small sweeps that mirror selected HPC workflows without claiming H100 performance |
| `environment_validation/` | Concrete `E001-E007` startup validation configs |
| `baseline_profiling/` | Concrete `E008-E019` single-GPU training and inference baselines |
| `workload_characterisation/` | Concrete `E020-E031` compute, memory, data pipeline, and inference profiler configs |
| `scaling_studies/` | Concrete `E032-E046` strong and weak scaling configs |
| `memory_optimisation/` | Concrete `E063-E082` checkpointing, micro-batch, and gradient accumulation configs |
| `precision_studies/` | Concrete `E083-E096` training and inference precision configs |
| `data_movement/` | Concrete `E097-E112` DataLoader, pinned memory, and prefetch configs |
| `communication_analysis/` | Concrete `E113-E120` NCCL/DDP and FSDP/NCCL communication profiling configs |
| `hardware_comparison/` | Concrete `E121-E128` H100 and L40S training/inference comparison configs |
| `local/` | Local smoke-test configs that do not require HPC GPUs |
| `training/` | Concrete training configs |
| `inference/` | Concrete inference configs, including framework and batching studies |
| `distributed/` | Concrete `E047-E062` DDP, FSDP, and DeepSpeed ZeRO strategy configs |
| `profiling/` | Profiling-focused configs and generated variants |

Concrete matrix experiment configs should preserve the matrix ID in both `experiment_id` and `metadata.matrix_id`. Local dev configs use readable `dev_` IDs and `metadata.profile: dev` so they stay separate from the `E001-E200` HPC tracker.
