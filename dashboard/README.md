# Research Dashboard — Experiment Catalog & plain-language summary

This is the published dashboard. The catalog now includes the verified L40S hardware-energy
anchors E235–E238 collected on 28 September 2026; the separate findings page remains a static
narrative snapshot through 27 September. The original `dashboard3/` files remain as the source
reference.

Two self-contained pages (no build step, no dependencies), meant to be read in order:

1. **`index.html` — Part 1, Experiment Catalog.** The complete 276-condition experiment matrix,
   grouped by research theme and study, with filters and an automatic "held constant vs. varied"
   breakdown per comparison, plus a click-through details view. The catalog overlays platform-specific
   CUDA OOM deferrals from `platforms/execution_matrix.json`: a condition may be complete on H100
   but **OOM-limited** at the tested settings on L40S. The catalog shows one compact platform-status
   column: green/blue/grey/red dots denote completed/configured/deferred/OOM-limited, with a legend,
   hover descriptions, and full details in the drawer. Platform is omitted from the varied chips and
   duplicate table columns. OOM-limited remains a distinct status filter and KPI;
   configured (not yet run) and other deferred conditions are not counted as measured OOMs.
   E250's first-attempt OOM is labelled recovered because its later canary completed. The overlay
   updates the displayed status counts without changing the embedded completed-trial means.
   The four hardware-energy anchors are synchronized from verified raw and processed evidence,
   including their measured-region energy, power, and reference experiment IDs. This is the evidence base.
2. **`analysis.html` — Part 2, Analysis & Findings.** Turns that evidence into a non-technical
   story for readers who aren't familiar with GPUs, distributed training, or the RQ1–RQ7 framing
   used elsewhere in this repo: six KPI numbers up top, then one short section per research
   question, each with a plain-English explanation, a simple chart, and a one-paragraph finding.

Both pages link to each other (a "Part 1 / Part 2" pill pair at the top of each, plus a
next/previous-part card at the bottom). Open either directly in a browser — nothing needs to be
served or built.

## Presentation approach

This site replaces the previous technical, CSV-driven dashboard. Its two pages are static and
self-contained, with the underlying experiment data embedded for offline reading.

`analysis.html` (Part 2) is a fixed narrative snapshot aimed at a non-technical reader: six KPI
numbers up top, then one short section per research question, each with a plain-English
explanation, a simple chart, and a one-paragraph finding. The underlying numbers are copied in
(rounded, with the analysis CSVs used to derive them noted below) rather than fetched live, so it
stays simple to open and share. `index.html` (Part 1) takes the opposite approach on purpose: it
shows the full, unrounded experiment matrix rather than a curated narrative subset.

Each finding card also links to a **"Supporting Evidence & Analysis"** section at the bottom of
the page, which is the deliberate exception to the plain-language rule: exact experiment IDs,
trial counts, comparability classes (`exact` / `platform_specific` / `capacity_adjusted`),
unrounded figures, and the source pipeline's own methodology/limitation notes, quoted rather than
paraphrased. It exists so a technical reader (a supervisor, an examiner, future-you) can verify
every rounded claim above against its exact source condition without leaving the page.

That section opens with three at-a-glance charts before the detail tables:
- a **coverage meter list** showing completed vs. planned test runs for every category that isn't
  yet 100% done (plus one summary row for the 12 categories that are),
- a **line chart** of scaling efficiency vs. chip count, one line per workload/scaling-type, and
- two small-multiple **scatter charts** (GPT-2 and DistilBERT) plotting throughput vs. memory per
  distributed-training strategy, with a ringed dot marking each workload's Pareto-frontier method.

All three are built with plain inline SVG (no chart library) and read their numbers from the same
figures quoted in the tables directly below them.

Every row in every "Supporting Evidence" table is also clickable (name, id cell, or anywhere in
the row) and opens a details drawer that slides in from the right, showing that experiment's full
condition (workload, strategy, precision, batch sizes, comparability class, status) and its
per-platform measured metrics (throughput, memory, latency, GPU utilization, power/energy,
AllReduce/NCCL, data-loading, as available), with `n` = repetition count. Clicking a row whose
source cell lists several experiment IDs (e.g. `E035, E036, E037`) opens one card per experiment.
An id suffixed `@h100`/`@l40s` filters the card to that platform only. An id with no completed
trial (shown as "(deferred)" in a table) still opens the drawer, with a note that it hasn't run
yet. Closes via the &times; button, clicking outside the drawer, or Escape; fully keyboard-operable
(tab to a row, Enter/Space to open).

The drawer's data lives in `experiment-details.json`, a static lookup keyed by experiment ID,
generated once from `results/processed/experiments.csv` (per-platform metric means) and
`results/processed/experiment_tracker.csv` (status/phase/completion date) &mdash; embedded directly
into `analysis.html` as the `EXPERIMENT_DETAILS` constant so the page still needs no server or
build step. It's kept alongside both pages for reference/regeneration but isn't fetched at
runtime. (Part 1's own, complete per-experiment dataset is embedded the same way in `index.html`,
and also kept alongside as `experiments-data.json`.)

## Regenerating it

The figures embedded in `analysis.html` (Part 2) were computed from these files as of the date in
the page's snapshot line:

- `results/analysis/research_question_summary.csv` and `research_finding_summary.csv` (RQ1/RQ2 scaling & strategy)
- `results/analysis/precision_summary.csv` and `results/analysis/trial_summary.csv` (RQ3 precision, matched by experiment ID)
- `results/analysis/communication_summary.csv` and `data_movement_summary.csv` (RQ4)
- `results/analysis/hardware_summary.csv` (RQ6)
- `results/analysis/inference_summary.csv` (batching)
- `results/processed/experiment_tracker.csv` (overall coverage numbers, and the `status`/phase
  columns behind the coverage meter list &mdash; 217 of 238 planned tests complete as of the last
  check, 91%)
- `results/processed/experiments.csv`, specifically the `metric_nvidia_smi_power_draw_watts_measured_region`
  and `metric_nvidia_smi_energy_joules_measured_region` columns (RQ5, the "Power & energy" section) —
  averaged per condition (n=3) and normalized to energy-per-1,000-tokens/responses by hand, since no
  pre-built summary CSV exposes that normalized figure directly

To refresh after a new experiment wave, re-run `python -m hpc_ai_perf.cli refresh`, re-derive the
handful of headline numbers from the files above, and update the constants inside the
`<script>` block in `analysis.html` (search for the `ready()` function — each chart call has its
values inline). There's no templating step; it's a static snapshot by design.

To refresh the details-drawer data, re-derive `experiment-details.json` from
`results/processed/experiments.csv` + `experiment_tracker.csv` (per experiment ID: workload/mode/
strategy/precision/batch metadata from the first matching row, plus per-platform means of
throughput, memory, latency, GPU utilization, `metric_nvidia_smi_power_draw_watts_measured_region`,
`metric_nvidia_smi_energy_joules_measured_region`, AllReduce time, NCCL bandwidth, and data-loading/
transfer time where present and non-zero), then paste the new JSON in as the `EXPERIMENT_DETAILS`
constant near the top of the `<script>` block in `analysis.html`.

To refresh Part 1's catalog after a new experiment wave, re-run the same two source files
(`experiment_tracker.csv` + `experiments.csv`) through the generation step described in the
2026-09-28 changelog entry below, and re-embed the result as the `DATA` constant near the top of
 the `<script>` block in `index.html` (also updating `experiments-data.json` alongside it). Keep the
 `CAPACITY_HISTORY` overlay aligned with the matrix's measured `cuda_oom` deferrals and its
 `recovered_failures` entry; `dashboard/site.test.cjs` checks that alignment.

For the four L40S hardware-energy anchors, run `python3 scripts/update_dashboard_energy_anchors.py`
after refreshing processed results. The script checks the raw evidence gate and trial provenance,
then synchronizes `experiments-data.json` and the embedded `DATA` in `index.html`. Use `--check`
to detect a stale catalog without changing files.

## Changelog

**2026-09-27** — Added a new "7B model (new)" section (`#lowbit`) plus matching
"Supporting Evidence" block (`#nerd-lowbit`), covering a 2026-09-26/27 evidence wave on
Qwen2.5-7B-Instruct (H100 only): serving speed/memory/quality across four number formats
(BF16/AWQ/GPTQ/NF4, E239-E246, E270-E273), long-context time-to-first-token (E247-E249), and
FSDP-vs-DeepSpeed-ZeRO-3 training throughput/memory at 2 and 4 GPUs (E250-E255). Also added two
KPI tiles to `#glance` and one sentence to the RQ1 (`#nerd-scaling`) limitation note about a new
capacity-adjusted probe (E256) that partially addresses the long-standing missing-E032 baseline
gap. 18 new entries were merged into `experiment-details.json` / `EXPERIMENT_DETAILS` (E239-E256),
bringing the total to 79.

Caveat carried into the page itself (see the "Still coming" card in `#lowbit`): this wave's
figures were read directly from raw trial JSON under `results/raw/h100/`, not from
`experiment_tracker.csv` / `experiments.csv`, because those processed files had not been
regenerated as of this update (`python -m hpc_ai_perf.cli refresh` needs a Python 3.11+
environment). The page-wide coverage KPI ("217 / 238") and the "Evidence coverage" meter list in
`#nerds` are therefore still pre-wave and do not yet count this batch of roughly three dozen new
completed tests. A third training-strategy arm (DDP, E261/E264/E266/E267) and the L40S side of
this same 7B model have no results yet either.

**2026-09-28** — Refreshed against the now-regenerated pipeline (`experiment_tracker.csv` /
`experiments.csv` were rebuilt 2026-09-27, resolving the Python-3.11 blocker noted above). Updated
the page-wide coverage KPI and "Evidence coverage" meter list from 217/238 to 250/276 (91%), broken
out by the current set of incomplete phases. Reworded the `#lowbit` "Still coming" card and the
`#nerd-lowbit` method note to drop the now-resolved pipeline-refresh caveat, while keeping the
still-real DDP-arm and L40S gaps. Added a new "Inference energy re-validation anchors" table to
`#nerd-power` covering three H100 DistilBERT FP16/BF16 conditions (E274-E276, completed
2026-09-26/27); merged their entries into `experiment-details.json` / `EXPERIMENT_DETAILS`
(now 82 entries) so their table rows open correctly in the details drawer instead of the "not in
the evidence set" fallback.

Investigated, but did **not** add: an apparent "H100 8-GPU scaling wave" surfaced by the pipeline
refresh (gpt2_small/distilbert/tinyllama at 8 GPUs in the refreshed `scaling_summary.csv`) turned
out on inspection to be **L40S** data (E216-E220), not H100 — the analysis pipeline groups by
platform, but a quick read of the summary rows alone doesn't show that. The H100-only
`#nerd-scaling` table was left as it was (still 1/2/4 GPUs only); the true H100 8-GPU wave and a
proper, clearly-labelled L40S comparison remain open items, not something to fold into an
H100-labelled table without a platform column.

**2026-09-28 (2)** — Added a finding-card to `#lowbit`, right after the existing FSDP-vs-ZeRO-3
speed finding, framing the same E252-E255 sharding data as a memory-capacity result: at 2 GPUs,
ZeRO-3's 77.44GB is within ~3GB of the H100's 80GB ceiling while FSDP's 62.11GB has real headroom.
This directly answers whether the 7B experiments contributed to the "does the model fit in memory"
line of inquiry — the pipeline's formal "Memory optimisation" phase never included this model (it's
GPT-2 Small/DistilBERT only, 25/31 completed), so this strategy comparison is the closest large-model
equivalent evidence, not a duplicate of it. Added a matching clarifying sentence to the
`#nerd-lowbit` methodology footnote for the same table.

**2026-09-29** — Added a new page, `experiments.html`, alongside the main report: a browsable catalog
of all 276 planned experiment conditions (not just the ~82 referenced in the main report's nerd-tables).
Built from `experiment_tracker.csv` (the full roster, including Configured/Deferred conditions with no
completed trial yet) enriched with per-trial results from `experiments.csv` (averaged across
repetitions, kept separate per platform).

Hierarchy: 10 research themes (Environment & Baselines, Scaling Efficiency, Distributed Strategy,
Precision & Quantization, Communication & Data Movement, Power & Energy, Hardware Comparison, Inference
Serving, Memory & Capacity, Large Model/Qwen2.5-7B) &rarr; 26 studies (the pipeline's own `phase` field,
unchanged) &rarr; one row-group per workload/model tested in that study. Each row-group shows two things
side by side: which parameters were held constant across every condition in it ("Held constant" chips)
and which were deliberately varied ("Varied" chips, plus matching table columns) &mdash; computed
automatically per group, not hand-curated, so it can't drift out of sync with the data.

Filters: theme, workload, platform, precision, strategy, status, mode, GPU count, plus free-text search;
filtering collapses/hides non-matching branches and auto-expands ones with matches. Clicking any row (or
a group's "Compare all" link) opens the same detail-drawer pattern as the main report, extended to show
every known parameter for that condition plus per-platform measured results (throughput, memory, energy,
latency, perplexity) where a trial has completed; Configured/Deferred conditions show their planned
parameters with an explicit "no completed trial yet" note instead of blank/misleading fields.

Linked from the main report's top nav ("Experiment Catalog") and from the Supporting Evidence section
intro. `experiments-data.json` is kept alongside as a plain data export of the same hierarchy (same
source the page embeds inline), in case it's useful to hand to a supervisor or reuse elsewhere.

Verification: full-coverage stress test (every one of the 276 experiment cards and all 52 workload
row-groups rendered through the actual page JS in Node with zero exceptions), tag-balance and CSS
brace-balance counts, `node --check` on the extracted script, and manual reconciliation of theme/phase
totals against `experiment_tracker.csv` (276/276, matches exactly). Live-browser click-through was not
possible this session (the same local-server/browser-tool instability noted in earlier entries), so this
is static + full-data-coverage verification, not a visual walkthrough.

**2026-09-29 (2)** — Reordered the two pages so the evidence comes before the narrative, and did a
publication pass over both. Renamed files: the plain-English report (formerly `index.html`) is now
`analysis.html`; the Experiment Catalog (formerly `experiments.html`) is now `index.html`, so it's
what opens by default. Both pages now carry a matching "Part 1 / Part 2" pill pair at the top (the
current page highlighted) and a next/previous-part card at the bottom, so the reading order is
explicit and either page can be opened first without getting lost. Updated every in-page
cross-link and this README's own file references to match.

Also tidied up language that read like a work-in-progress lab notebook rather than a finished
document: the 7B-model section's kicker changed from the date-stamped "New evidence · 2026-09-26"
to "Question 8" (continuing the plain-English page's existing Question 1–7 numbering — this is a
reading-order label only, not a claim about the project's formal RQ1–RQ7 taxonomy, which the 7B
work sits across several of rather than owning a number of its own), its nav link dropped the
"(new)" suffix, and its Supporting Evidence heading dropped the "New evidence (2026-09-26/27)"
prefix (the dates still appear in that block's methodology paragraph, where they're provenance
rather than a headline). No figures, findings, or caveats were changed — this was navigation and
framing only.
