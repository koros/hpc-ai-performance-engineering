# Local Dev Runbook

Use the dev profile to exercise baseline-shaped runs on a local Mac when HPC is unavailable. These runs are useful for checking workload code, model loading, result writing, aggregation, and dashboard plumbing. They are not H100 performance results and should not be used as dissertation measurements.

For the rationale, parameter choices, and the local-dev versus HPC-only matrix, see [`docs/dev_experiment_matrix.md`](dev_experiment_matrix.md).

## Profile Boundaries

| Profile | Purpose | Config path | Result meaning |
| --- | --- | --- | --- |
| Smoke | Minimal plumbing checks | `configs/local/` | Confirms the smallest runnable paths work. |
| Dev | Lightweight local baseline-style runs and small sweeps | `configs/dev/` | Confirms baseline, memory-shape, inference-length, and synthetic workflows on Mac/CPU/MPS. |
| HPC | Real Lovelace experiments | `configs/*/E*.json` | Dissertation experiment measurements. |

## Setup

Install the local workload dependencies:

```bash
python -m pip install -e ".[dev,yaml,workloads]"
```

The dev configs use `device: auto`. The workload runner selects CUDA if available, then Apple MPS if available, then CPU. To force CPU for a specific run, edit the config parameter:

```json
"device": "cpu"
```

## Run Dev Experiments

Run only the baseline-shaped dev configs:

```bash
make run-dev-baselines
```

Run all dev configs, including small sweeps:

```bash
make run-dev-experiments
```

Run one config directly:

```bash
python -m hpc_ai_perf.cli run --config configs/dev/dev_gpt2_small_training_baseline.json
```

Refresh processed outputs after the runs:

```bash
python -m hpc_ai_perf.cli refresh
make track-dev-progress
```

## Inspect Results

Raw JSON results are written to:

```text
results/raw/dev_*.json
```

Processed outputs are written to:

```text
results/processed/experiments.csv
results/processed/dev_experiment_tracker.csv
results/processed/dev_experiment_tracker.md
results/analysis/*.csv
```

Interpret `dev_` results separately. The default `experiment_tracker.*` files remain the real HPC matrix tracker, including separately labelled supplemental platform probes when present; the `dev_experiment_tracker.*` files summarize local Mac readiness only.
