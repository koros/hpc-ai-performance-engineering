# Loading the same data in R and JavaScript

`qwen_energy_concurrency.R` reads the CSV and creates a ggplot figure; run it from
the project root. It requires readr, dplyr and ggplot2. File saving is an explicit,
commented step so the example does not overwrite figures.

`qwen_energy_concurrency.mjs` reads the matching JSON and returns Plotly-compatible
traces/layout. It does not require Plotly to load/transform the data; supply your
preferred Plotly build in the calling page. The same flat rows can be mapped to
Chart.js datasets or Vega-Lite data values. Serve the dataset folder over HTTP.

Both examples separate recorded runtime identities, use the exported confidence
bounds and show measured points without implying an interpolated threshold.
Neither pools canaries or request samples with independent performance trials.

## Generate four ready-to-use figures with base R

From the repository root:

```sh
Rscript scripts/plot_study_results.R
```

No R packages are required. The script reads the exported CSV files and writes
PNG and vector PDF versions to `results/figures/r_study_review/`. An optional
first argument selects another output directory. Regenerate missing or stale
input tables with `make export-plot-data PYTHON=python` first.

Figures cover Qwen7B energy, throughput and first-token latency versus concurrency,
and training scaling efficiency for GPT-2 Small, DistilBERT and TinyLlama.
Qwen comparisons fix eight L40S GPUs, 1,280 prompt tokens and 128 output tokens.
Training curves hold per-GPU workload fixed; global batch increases, so these
are not fixed-global-batch strong-scaling curves or convergence comparisons.

Points use condition means and exported approximate trial-level 95% intervals.
The script checks expected condition counts, three repeats and runtime separation.
Concurrency 2 is explicitly marked as a follow-up motivated by observed results.
Selected plot data, input checksums, runtime identities and R session information
are saved alongside the figures. Generated artifacts remain outside Git under
the existing figures retention policy.
