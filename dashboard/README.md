# Dashboard

Static dashboard for inspecting processed experiment outputs.

Expected inputs:

- `results/processed/experiments.csv`
- `results/analysis/phase_summary.csv`
- `results/analysis/trial_summary.csv`
- `results/analysis/experiment_rankings.csv`
- `results/analysis/scaling_summary.csv`
- `results/analysis/precision_summary.csv`
- `results/analysis/strategy_summary.csv`
- `results/analysis/communication_summary.csv`
- `results/analysis/data_movement_summary.csv`
- `results/analysis/memory_summary.csv`
- `results/analysis/inference_summary.csv`
- `results/analysis/hardware_summary.csv`
- `results/processed/readiness_report.md`

The dashboard loads these files automatically when served from the repository root. The `Analysis CSVs` file picker
also accepts several analysis files at once for an offline/manual review.

## Presentation rules

- L40S and H100 measurements are never averaged into one chart series.
- `exact`, `platform_specific`, and `capacity_adjusted` results remain separate.
- Throughput charts use separate scales for different throughput units.
- Empty study panels are removed from the main flow and listed under **Awaiting evidence** at the bottom.
- Narrow screens render result tables as labelled cards, avoiding page-wide horizontal scrolling.
- The metric-focus filter can reduce the page to throughput, runtime, memory, energy/power, or latency views.

Run the pipeline and serve the dashboard locally:

```bash
python -m hpc_ai_perf.cli refresh
make serve-dashboard
```

Open `http://localhost:8000/dashboard/`.
