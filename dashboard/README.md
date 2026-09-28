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
- `results/analysis/lowbit_summary.csv`
- `results/analysis/hardware_summary.csv`
- `results/analysis/research_question_summary.csv`
- `results/analysis/research_finding_summary.csv`
- `results/processed/readiness_report.md`

The dashboard loads these files automatically when served from the repository root. The `Analysis CSVs` file picker
also accepts several analysis files at once for an offline/manual review.

## Presentation rules

The front page now starts with a short, live-data research synthesis. Detailed study
charts, coverage cards, and source-level findings sit in the closed **Explore the
supporting studies** section. The trial audit is a separate disclosure. This keeps
one scaling plot and one strategy comparison in the technical explorer instead of
repeating both immediately beneath their conclusions. Each takeaway has an
expandable explanation of the changed variable, matched controls, and limitation;
source conditions remain clickable. The low-bit and long-context conclusions are
derived from their refreshed analysis CSVs, not copied from `dashboard3`'s fixed
snapshot.

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

- The **current evidence snapshot** reports active-view conditions, validated
  trials, research-question readiness, validated FP8 coverage, measured-region
  energy coverage, and the remaining legacy-telemetry boundary.
- The **campaign synthesis** leads with evidence-derived conclusions
  rather than asking readers to infer them from charts. Each card reports an
  effect size, decision implication, changed variable, held-constant controls,
  limitation, and clickable source conditions. It currently covers scaling,
  distributed strategy, backend-matched FP8, checkpointing, NCCL capability,
  matched H100/L40S performance, the remaining hardware-energy gap, 7B
  low-bit serving, and H100 long-context latency.
- **RQ1–RQ7 decision cards** condense platform-level coverage into one card per
  research question. Each card shows trial completion, platform scope, the
  remaining limitation, and a direct link to the relevant evidence view.
- **RQ1 scaling** uses a single chart in the supporting-studies explorer. Its
  platform and scaling-protocol series remain separate.
- **RQ2 strategy trade-offs** use one matched-comparison selector in the
  explorer; unlike workload, model, precision, GPU-count, and batch setups are
  never put on the same axis.
- **Traceable findings** link every displayed RQ1/RQ2 statistical finding to
  its source condition IDs and trial sufficiency. The campaign synthesis adds
  controlled RQ3–RQ6 interpretations and a scoped RQ5 limitation without
  replacing those underlying analysis tables.
- **Controlled Study Map** provides the bridge from collected phases to the
  research views. It prevents workload, hardware, strategy, scaling, precision,
  and memory changes from being compressed into one confounded leaderboard.
- **RQ3 precision controls** now include validated FP8 training alongside FP32,
  FP16, and BF16, with throughput, memory, and short-run quality-loss context.
- **RQ6 hardware comparisons** match H100 and L40S by protocol rather than by
  experiment ID. Throughput is shown beside memory for training or p50 latency
  for inference; unmatched or legacy energy measurements are not implied.

The five most commonly used filters remain visible. Comparability, phase,
strategy, precision, GPU count, wave, and evidence-quality controls are grouped
under **More filters** to keep the initial view focused on interpretation.

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
