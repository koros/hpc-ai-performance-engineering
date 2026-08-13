# Experiments

The experiment workspace mirrors the dissertation proposal outline and the latest Appendix A matrix. Each phase folder is for scripts, notes, generated configuration variants, and phase-specific analysis helpers.

| Phase folder | Proposal section | Matrix entries |
| --- | --- | --- |
| `environment_validation/` | Environment Validation | E001-E007 |
| `benchmark_selection/` | Benchmark Selection | Benchmark catalogue |
| `baseline_profiling/` | Baseline Profiling | E008-E019 |
| `workload_characterisation/` | Workload Characterisation | E020-E031 |
| `precision_studies/` | Precision Studies | E083-E096, E149-E200 |
| `scaling_studies/` | Scaling Studies | E032-E046 |
| `distributed_training_strategies/` | Distributed Training Strategies | E047-E062 |
| `memory_optimisation/` | Memory Optimisation | E063-E082 |
| `data_movement_communication/` | Data Movement and Communication | E097-E120 |
| `hardware_comparison/` | Hardware Comparison | E121-E128 |
| `inference_studies/` | Inference Studies | E129-E200 |
| `integrated_analysis_dashboard/` | Integrated Analysis and Dashboard | A01-A07 |
| `recommendations/` | Recommendations | A08 |

Concrete runnable configs live under `configs/`, while this `experiments/` tree is for notes, helper scripts, and phase-specific analysis artefacts. The current generated matrix status is written to `results/processed/experiment_tracker.md` by:

```bash
python -m hpc_ai_perf.cli refresh
```

Local Mac/workstation dev runs are tracked separately with:

```bash
make track-dev-progress
```
