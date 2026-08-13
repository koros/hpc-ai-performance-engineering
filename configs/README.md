# Experiment Configs

This directory contains runnable experiment configs and reusable templates.

| Path | Purpose |
| --- | --- |
| `baseline_profiling/` | Concrete `E008-E019` single-GPU training and inference baselines |
| `communication_analysis/` | Concrete `E113-E120` NCCL/DDP and FSDP/NCCL communication profiling configs |
| `data_movement/` | Concrete `E097-E112` DataLoader, pinned memory, and prefetch configs |
| `dev/` | 15 local Mac/workstation development baselines and small sweeps, tracked separately from HPC results |
| `distributed/` | Concrete `E047-E062` DDP, FSDP, and DeepSpeed ZeRO strategy configs |
| `environment_validation/` | Concrete `E001-E007` startup validation configs |
| `hardware_comparison/` | Concrete `E121-E128` H100 and L40S training/inference comparison configs |
| `inference/` | Concrete `E129-E200` inference configs, including framework, batching, concurrency, prompt length, multi-GPU, and precision studies |
| `local/` | Local smoke-test configs that do not require HPC GPUs |
| `memory_optimisation/` | Concrete `E063-E082` checkpointing, micro-batch, and gradient accumulation configs |
| `precision_studies/` | Concrete `E083-E096` training precision configs |
| `profiling/` | Profiling-focused configs and generated variants |
| `scaling_studies/` | Concrete `E032-E046` strong and weak scaling configs |
| `templates/` | Valid reusable JSON configs for each experiment family |
| `training/` | Additional concrete training configs not assigned to the main matrix phases |
| `workload_characterisation/` | Concrete `E020-E031` compute, memory, data pipeline, and inference profiler configs |

Concrete matrix experiment configs should preserve the matrix ID in both `experiment_id` and `metadata.matrix_id`. Local dev configs use readable `dev_` IDs and `metadata.profile: dev` so they stay separate from the `E001-E200` HPC tracker.
