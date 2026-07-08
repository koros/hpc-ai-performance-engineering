# Dashboard

Static dashboard for inspecting processed experiment outputs.

Expected inputs:

- `results/processed/experiments.csv`
- `results/analysis/phase_summary.csv`

Run the pipeline and serve the dashboard locally:

```bash
python -m hpc_ai_perf.cli aggregate-results
python -m hpc_ai_perf.cli analyze-results
make serve-dashboard
```

Open `http://localhost:8000/dashboard/`.
