# Dashboard

Static dashboard for inspecting processed experiment outputs.

Expected inputs:

- `results/processed/experiments.csv`
- `results/analysis/phase_summary.csv`
- `results/analysis/experiment_rankings.csv`
- `results/analysis/scaling_summary.csv`
- `results/analysis/precision_summary.csv`
- `results/processed/readiness_report.md`

Run the pipeline and serve the dashboard locally:

```bash
python -m hpc_ai_perf.cli refresh
make serve-dashboard
```

Open `http://localhost:8000/dashboard/`.
