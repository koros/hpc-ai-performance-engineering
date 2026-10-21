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
