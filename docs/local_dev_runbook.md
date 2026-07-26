# Local Dev Runbook

Use the dev profile when HPC is unavailable and you want to exercise baseline-shaped runs on a local Mac. These runs are useful for checking workload code, model loading, result writing, aggregation, and dashboard plumbing. They are not H100 performance results and should not be used as dissertation measurements.

## Profile Boundaries

| Profile | Purpose | Config path | Result meaning |
| --- | --- | --- | --- |
| Smoke | Minimal plumbing checks | `configs/local/` | Confirms the smallest runnable paths work. |
| Dev | Lightweight local baseline-style runs | `configs/dev/` | Confirms baseline workflows on Mac/CPU/MPS. |
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

## Run Dev Baselines

Run all dev configs:

```bash
make run-dev-baselines
```

Run one config directly:

```bash
python -m hpc_ai_perf.cli run --config configs/dev/dev_gpt2_small_training_baseline.json
```

Refresh processed outputs after the runs:

```bash
python -m hpc_ai_perf.cli refresh
```

## Inspect Results

Raw JSON results are written to:

```text
results/raw/dev_*.json
```

Processed outputs are written to:

```text
results/processed/experiments.csv
results/processed/experiment_tracker.csv
results/processed/experiment_tracker.md
results/analysis/*.csv
```

Keep `dev_` results separate in your interpretation. They validate workflow readiness; they do not replace Lovelace H100 or L40S measurements.
