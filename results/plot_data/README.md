# Plot-ready research data for R and JavaScript

Regenerate from the repository root:

```bash
make export-plot-data PYTHON=python
```

Every dataset has matching **CSV and JSON** files. CSV is rectangular and works
with R/readr, pandas, Excel, D3 and Observable. JSON is an array of flat records
with real numbers/booleans and `null` for missing values; it works with Plotly,
Chart.js, Vega-Lite and other JavaScript libraries. No `NaN` or `Infinity` is emitted.
The source-of-truth files in `results/raw` are never modified.

## Folder purposes

| Folder | Main datasets | Use |
|---|---|---|
| `trial_level/` | `all_recorded_trials`, `current_performance_trials`, `performance_condition_summary` | Audit individual repetitions; calculate or inspect condition means and uncertainty |
| `training_scaling/` | `deep_dive_condition_metrics`, `scaling_efficiency_curves` | Throughput, GPU-count efficiency curves and memory use for the controlled training deep dive |
| `training_parameter_studies/` | Accumulation, batch/sequence, model size, DDP settings and distributed-strategy condition tables | Facets, heatmaps and within-study comparisons; controls may live in the scaling table |
| `training_parameter_effects/` | `all_controlled_effects` and one table per measured lever | Treatment/control forest plots and percentage-effect comparisons |
| `inference_layout_comparisons/` | `tinyllama_conditions`, `qwen7b_conditions` | Fixed-GPU-budget TP/replica layout comparisons |
| `inference_concurrency/` | `qwen7b_long_prompt_conditions` | Qwen TP2/TP4 demand-dependent throughput, latency and energy rankings |
| `energy_latency_tradeoffs/` | `inference_condition_metrics`, `training_condition_metrics` | Throughput–energy–latency scatter plots and Pareto-style exploration |
| `request_distributions/` | `inference_samples_long` | Within-trial ECDFs, boxplots and violin plots for the recent inference campaigns |
| `matrix_coverage/` | `platform_condition_coverage` | Completion/status plots, separately for each platform |
| `provenance/` | `source_records` | Trace any observation to its raw file, checksum and selected cohort |
| `examples/` | R and JavaScript examples | Load and plot the Qwen energy/concurrency comparison |

`manifest.json` lists all dataset paths, row counts, column schemas, units and
output checksums. `data_dictionary.json` provides the machine-readable schemas;
[DATA_DICTIONARY.md](DATA_DICTIONARY.md) explains the analysis conventions.
The generated CSV/JSON outputs are ignored in Git, consistent with the project's
result-retention policy. The generator, examples and documentation are tracked.
Rebuild the data after pulling/collecting new evidence. A fresh checkout will have
the folder documentation but needs the command above to populate its datasets.

## Selection and aggregation

- All recorded H100/L40S trials, including failures and canaries, remain in the
  audit table. Raw schema/checksum validation must pass before exporting.
- Analysis selects the **latest submission cohort per experiment/platform**.
  Config hashes and recorded runtime fingerprints/versions distinguish cohorts.
  Failed latest attempts do not silently fall back to an older successful attempt.
- Default summaries include completed, error-free performance trials from that
  cohort. Explicitly dirty source runs are excluded. Canaries/diagnostics are
  available in the audit table but are not performance evidence.
- Model, precision, request shape and platform are not averaged together.
  Generic summaries have one row per experiment/platform/cohort. Training scaling
  and effect tables additionally match baselines inside recorded runtime cohorts.
- Runtime identity uses the recorded fingerprints/versions, with a commit fallback
  when no fingerprint exists. It cannot recover unrecorded library/build changes.
- Means are unweighted across independent repetitions. Each metric has its own
  available-value count. Approximate Student-t 95% intervals use the existing
  analysis critical-value lookup (normal approximation above 30 degrees of freedom).
  A single repetition has **null** summary confidence bounds, not a zero-width CI.
- Missing measurements remain missing, rather than zero. No energy is extrapolated
  from startup/full-window totals. Training J/token and inference J/output token
  are different measures and must not be pooled.
- The request table is **long form**: one row per sample/metric/trial. Its index is
  not a cross-metric request identifier. Requests within a trial are not independent
  experiment repetitions; do not use them to inflate between-trial sample size.
- Latency summary means average trial-level p95s. They are not pooled-request p95s.
- The concurrency-two conditions retain **“follow-up motivated by observed results”**
  and their motivating experiment IDs. Keep this distinction in report captions.

## Loading

R, from the repository root:

```r
library(readr)
x <- read_csv("results/plot_data/inference_concurrency/qwen7b_long_prompt_conditions.csv")
```

JavaScript, from a page served with `results/plot_data/` as its root:

```js
const rows = await fetch("inference_concurrency/qwen7b_long_prompt_conditions.json")
  .then(response => response.json());
```

Use a local HTTP server for browser `fetch`; opening JSON-fetching pages directly
with `file://` can be blocked by the browser. Serve the exports alone with:
`python -m http.server 8000 --directory results/plot_data`.
