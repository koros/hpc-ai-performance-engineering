# Current Project Status

This snapshot summarises the repository state before full HPC submission. It should be updated after major workflow or experiment changes.

## Ready Now

| Area | Status |
| --- | --- |
| Experiment matrix | 200 concrete matrix configs are present and load successfully. |
| Validation | `python -m hpc_ai_perf.cli validate --skip-results` passes with 0 config issues. |
| Local smoke runner | CPU-only synthetic workload is ready and covered by tests. |
| Result pipeline | Raw JSON aggregation, analysis tables, progress tracker, readiness report, and dashboard refresh are wired into `python -m hpc_ai_perf.cli refresh`. |
| Slurm support | Single-config rendering, filtered bulk rendering, and three transformer smoke presets are available. |
| Dashboard | Static dashboard reads processed results, analysis summaries, and readiness report output. |
| Quality gates | `pytest`, `ruff`, and `pre-commit` are configured and passing. |

## Smoke-Ready Workloads

| Workload | Current execution support | Smoke config | Notes |
| --- | --- | --- | --- |
| `synthetic` | Ready | `configs/local/synthetic_smoke.json` | Runs without GPU or optional ML dependencies. |
| `gpt2_small` | Optional training and inference smoke path | `configs/local/gpt2_small_smoke.json` | Requires `pip install -e ".[workloads]"`. |
| `distilbert` | Optional training and inference smoke path | `configs/local/distilbert_smoke.json` | Requires `pip install -e ".[workloads]"`. |
| `tinyllama` | Optional inference smoke path | `configs/local/tinyllama_inference_smoke.json` | Training is intentionally unsupported for now. |

## Current Matrix Breakdown

| Category | Count |
| --- | ---: |
| Total matrix configs | 200 |
| Must Have | 116 |
| Should Have | 71 |
| Optional | 13 |
| Training configs | 107 |
| Inference configs | 93 |

## Before Full HPC Submission

1. Install the workload extra on the cluster:

   ```bash
   python -m pip install -e ".[dev,yaml,workloads]"
   ```

2. Edit Slurm smoke presets for partition, account, modules, and virtual environment:

   ```bash
   slurm/smoke/gpt2_small_smoke.sbatch
   slurm/smoke/distilbert_smoke.sbatch
   slurm/smoke/tinyllama_inference_smoke.sbatch
   ```

3. Run the HPC smoke checklist:

   ```bash
   docs/hpc_smoke_runbook.md
   ```

4. Refresh generated outputs:

   ```bash
   python -m hpc_ai_perf.cli refresh
   ```

5. Review:

   - `results/processed/readiness_report.md`
   - `results/processed/experiment_tracker.md`
   - dashboard at `http://localhost:8000/dashboard/`

## Main Remaining Work

| Area | Remaining work |
| --- | --- |
| HPC environment | Run and confirm the three Slurm smoke jobs on Lovelace. |
| Real benchmark execution | Submit Must Have matrix batches after smoke jobs pass. |
| Profiling | Add real GPU profiler adapters beyond the current null/local path. |
| Distributed execution | Expand runner support for DDP/FSDP/ZeRO orchestration beyond config and Slurm generation. |
| Metrics | Add power/energy and deeper communication metrics when platform tooling is confirmed. |
| Dashboard | Add charts for scaling efficiency, precision comparisons, and hardware comparison once results exist. |
