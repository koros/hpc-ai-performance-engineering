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
- `results/analysis/research_question_summary.csv`
- `results/analysis/research_finding_summary.csv`
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
- Every analytical chart states which field or fields vary and lists a control only when that value is identical across the plotted rows.
- The training-performance view shows one study at a time. Its dot is mean throughput, its whisker is the 95% confidence interval, and its run-variation percentage is the sample standard deviation divided by the mean. Strategy and precision comparisons use matched controls and report change from DDP or BF16 respectively; groups with only one configuration stay out of the comparison view.
- The distributed-strategy chart shows one matched platform, workload, precision, and GPU-count comparison at a time. A selector switches among independent comparisons, a metric selector switches between throughput and measured GPU memory, and stable colours identify DDP, FSDP, and each ZeRO stage without implying that unrelated groups share a scale.
- The precision view shows one matched workload at a time and uses coordinated throughput and measured-memory charts. BF16, FP16, and FP32 retain the same colour in both charts, while the summary and takeaway quantify BF16 relative to FP32. The current view therefore communicates the available L40S evidence without implying equivalent H100 precision coverage.
- The overview groups completed evidence into controlled study families, names the variable tested, and links to the appropriate research panel. Condition and trial totals indicate coverage only; they are never presented as performance rankings.
- Memory evidence is routed by `study_type`: micro-batch and gradient-accumulation sweeps use their respective numeric setting on the x-axis, while activation checkpointing uses a controlled off/on comparison. These studies are not combined into one apparent micro-batch curve.
- Data-movement evidence is likewise separated by its independent variable: worker count and prefetch depth use millisecond-scale curves, while pinned memory uses an off/on comparison with loading-time and throughput deltas. Platform, workload, and the remaining loader settings stay fixed within each panel.
- Cross-phase throughput is kept in the evidence section as a descriptive distribution. Trials are averaged within each condition first, and each condition then contributes once to the displayed median and minimum–maximum range. The phase inventory intentionally omits pooled throughput averages.
- Active filters appear as removable chips; **Reset filters** restores the complete evidence view.

## Experiment drill-down

Select an experiment row, bar, or chart point to open its condition-level detail view. The dialog keeps trial evidence together while separating:

- protocol and recorded controls;
- performance and memory;
- measured GPU utilisation, power, and energy;
- Torch-profiler CPU time and data-movement timing;
- inference latency and distributed-communication metrics;
- trial consistency, environment details, and provenance.

Metric groups with no recorded values are omitted. The trial mini-chart uses aggregate values from each trial; it does not imply that telemetry sample histories were captured. “Torch profiler CPU time” is not presented as host CPU utilisation because the current result schema does not record that metric.

## Research-question views

The first dashboard section turns refreshed evidence into an explicit research
argument rather than a generic results explorer:

- **RQ1–RQ7 coverage cards** show direct condition/trial coverage, platform and
  comparability scope, and the limitation that applies to each question.
- **RQ1 scaling** plots only within-platform speedup and parallel efficiency.
  L40S scaling is labelled supporting evidence because the primary question is
  framed around H100 one-to-four-GPU scaling.
- **RQ2 strategy trade-offs** plot maximum observed GPU memory against mean
  throughput only where workload, model, precision, GPU count, sequence length,
  global batch, and accumulation settings match. The observed Pareto frontier is
  shown instead of claiming a universal best strategy.
- **Traceable findings** link every displayed RQ1/RQ2 statement to its source
  condition IDs and trial sufficiency. RQ3–RQ7 remain visible as coverage cards
  until their dedicated conclusion views are added.
- **Controlled Study Map** provides the bridge from collected phases to the
  research views. It prevents workload, hardware, strategy, scaling, precision,
  and memory changes from being compressed into one confounded leaderboard.

The Research question filter applies to raw trial rows and the research-specific
tables. Other summary tables intentionally ask the user to clear that filter,
because they do not contain an RQ-level grouping and must not be reinterpreted
as such.

Run the pipeline and serve the dashboard locally:

```bash
python -m hpc_ai_perf.cli refresh
make serve-dashboard
```

Open `http://localhost:8000/dashboard/`.

## GitHub Pages

The `Deploy research dashboard` workflow rebuilds all derived summaries from committed raw evidence, runs the project
and dashboard checks, and publishes a static artifact to the public
[`hpc-ai-performance-engineering-dashboard`](https://github.com/koros/hpc-ai-performance-engineering-dashboard)
repository whenever `main` changes. The artifact contains only the dashboard, analysis CSVs, the processed experiment
table, and the readiness report; raw evidence and local build files are excluded.

The separate public repository is a transitional host while the complete research repository remains private. Once this
repository completes its open-source review and becomes public, Pages can be moved here without changing the dashboard
build process. The hosted URL is <https://koros.github.io/hpc-ai-performance-engineering-dashboard/>.

To reproduce the publication artifact locally:

```bash
make build-dashboard-site
```

The generated site is written to `build/pages/`.
