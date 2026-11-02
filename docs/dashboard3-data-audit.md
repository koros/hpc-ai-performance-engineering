# Part 3 correction and integration — 2 November 2026

## Current outcome

The corrected Part 3 is now in `dashboard/studies.html` and
`dashboard/studies-data.json`, linked from the current catalog and findings.
It contains 17 studies, 409 distinct experiment IDs and **1,474 unique completed
trials**. The original dashboard3 prototype has not been promoted wholesale.
The earlier hold decision below is superseded for this corrected Part 3.

## Corrections and verification

- Replaced the sparse Student-t lookup in upstream deep-dive/plot calculations
  and the current study generator with SciPy's numerical 0.975 quantile. Welch
  degrees of freedom remain fractional. SciPy is declared in `pyproject.toml`.
- Regenerated all plot exports from checksum-verified raw evidence: 3,139 source
  records and 2,553 current performance trials. Comparing the old and corrected
  CSVs found no changes outside confidence bounds and the resulting verdict.
  Measurements, means, cohort membership and provenance identities are unchanged.
- Independently recomputed all 270 unique controlled effects from current trial
  throughputs, using Welch variances and `scipy.stats.t.ppf`. All effect means and
  all 269 available intervals agree within export rounding. The remaining effect
  has insufficient repetitions and retains an inconclusive verdict.
- Exactly one unique verdict changes: L40S DistilBERT at 4 GPUs, E404 → E589,
  DDP versus grad-op FSDP: **+5.100379%, interval −1.740580% to +11.941339%,
  inconclusive**. The study table and embedded offline dataset carry this result.
- Report-based throughput charts now take intervals from the refreshed current
  trial summaries. Scaling, precision and run-length ratio intervals are
  recalculated with numerical Student-t quantiles. Reference conditions compared
  with themselves have ratio 1 and bounds 1.
- Goodput means, SDs and confidence bounds are recomputed from individual request
  latencies and each trial's measured duration. Missing intervals remain missing.
- The page explains small repetition counts and the limits of effect/ratio
  intervals. The S5 prose now limits its per-layer-wrapping claim to the supported
  8-GPU contrasts instead of implying every smaller-GPU contrast is conclusive.
- `make refresh` and `make refresh-dashboard` regenerate plot evidence before
  rebuilding the studies, preventing reuse of the old interval exports.

The preceding review independently matched 11,928 metric means, 950 report metric
fields, 53 derived ratios and 510 goodput means. This correction preserves all
underlying measurements and means. Statistical intervals describe repeated-trial
variation under the implemented method; they do not establish time-to-accuracy
or account for every experimental uncertainty.

Reproduction (after installing the project with `python -m pip install -e ".[dev]"`):

```bash
make refresh-dashboard PYTHON=python
make check-dashboard-data PYTHON=python
python -m pytest tests/test_dashboard_studies.py tests/test_plot_data.py tests/test_scaling_deep_dive.py tests/test_build_dashboard_site.py tests/test_dashboard_results.py
node dashboard/site.test.cjs
python scripts/build_dashboard_site.py --output-dir build/pages
```

The JavaScript checks execute all 17 study renderers, charts and tables, validate
the unique-trial headline and search filtering, and reject NaN/undefined/infinite
output. Visual browser inspection could not be completed: the in-app browser
could not reach localhost and native Chrome automation timed out.

---

# Dashboard3 review — 2 November 2026

## Current decision

Do not replace `dashboard/` with `dashboard3/` as it stands. The source
measurements reconcile, but the trial headline and confidence-interval method
still prevent an unqualified data-quality approval. No dashboard files were
changed during this review.

Fresh verification against the current repository:

- Generator `--check` passes.
- Raw-evidence export verifies 3,139 records and 2,553 current performance trials.
- All regenerated CSV fields match the checked-in plot exports, including
  provenance fields. The older checksum finding below is now resolved.
- 11,928 independently computed metric means across 852 conditions, 950 report
  metric comparisons, 53 derived ratios and 510 goodput points have no mismatches.
- Part 3 still counts 1,663 trial appearances instead of 1,474 unique trials
  across 409 experiment IDs: an overstatement of 189 (12.8%).
- `_t95` still rounds Welch degrees of freedom upward to a sparse table entry,
  narrowing intervals. The earlier exact-t recalculation below documents the
  affected E404 → E589 verdict; that exact-t calculation was not rerun in this
  review because SciPy is unavailable in the current Python environment.
- Part 2 retains its explicitly labelled September narrative snapshot.

The commands in the reproduction section below were rerun. Fresh numerical
results are in `build/dashboard3-audit/audit-results.json`. The previous review
is retained below as historical context; its provenance warning is superseded
by the fresh field-for-field export comparison above.

---

# Dashboard3 data audit — 28 October 2026

## Verdict

Dashboard3 uses the intended evidence sources, and the numerical aggregates checked
against current raw evidence are correct. It is **not fully accurate as presented**:
the Part 3 trial KPI double-counts overlapping studies, the upstream approximate
confidence-interval calculation changes one statistical verdict, and the checked-in
plot exports retain provenance identities from before the public-data sanitization.

This audit did not change dashboard3, raw evidence, or the upstream exports.

## Checks and results

| Check | Result |
| --- | --- |
| Dashboard3 generator `--check` | Passed: embedded/generated outputs match its existing inputs |
| Fresh export from raw evidence into a separate audit directory | Passed evidence verification; 3,139 records and 2,553 current performance trials |
| Independent trial means | 11,928 metric means across 852 platform/condition groups; zero mismatches |
| Cohort separation within those condition groups | One config/runtime/submission identity per group |
| Report CSV metrics versus fresh condition summaries | 950 numeric comparisons across eight report tables; zero mismatches |
| Derived scaling, per-GPU and hardware ratios | 53 comparisons; zero mismatches |
| Goodput from individual request latencies and measured trial durations | All 510 mean points reproduced |
| Controlled treatment/control mean changes | All 270 reproduced |
| Catalog condition IDs | All 817 configurations represented, without missing or extra IDs |
| Catalog status counts | 746 completed, 2 partial, 48 configured, 21 deferred |

A catalog condition marked completed means it completed on at least one routed
platform. It does not establish that both platform arms are complete. The
platform-specific coverage fields retain that distinction.

## Corrections needed

### 1. The trial KPI counts study appearances, rather than unique trials

`dashboard3/studies.html` adds each study's coverage trial count. Some experiments
support more than one study, so the result is **1,663** trial appearances, compared
with **1,474 unique completed trials** across the **409 distinct experiment IDs**.
The headline is inflated by 189 (12.8%). Deduplicate by experiment/platform, or by
run ID for a direct trial count, before producing the page-wide total.

This issue was already corrected in the new `dashboard/studies.html`; the original
dashboard3 prototype still has it.

### 2. Approximate Student-t critical values can overstate confidence

`src/hpc_ai_perf/analysis/scaling_deep_dive.py::_t95` rounds fractional Welch degrees
of freedom upward to a sparse lookup-table entry. This can choose a smaller critical
value than required and narrow the confidence interval.

I recalculated the intervals for the 269 contrasts with sufficient repetitions,
using SciPy's `t.ppf(0.975, df)` and the same Welch difference-of-means method used
by the pipeline. One of the 270 contrasts has insufficient repetitions for an
interval. All mean percentage changes agree; one verdict changes:

| Comparison | Dashboard3 | Recalculated with exact t critical value |
| --- | --- | --- |
| L40S DistilBERT, 4 GPUs, E404 → E589; DDP versus FSDP grad-op shard | +5.10%; 95% CI +0.03% to +10.17%; **helps** | +5.10%; 95% CI −1.74% to +11.94%; **inconclusive** |

This is an upstream calculation issue, rather than a transcription error in
dashboard3. The newly added dashboard studies consume the same upstream effects,
so they inherit this issue too. Correct the critical-value calculation and
regenerate the effect exports before treating all verdicts as verified.
The recalculation checks the implemented statistical method; it does not establish
that the method captures every source of experimental uncertainty.

### 3. Provenance hashes need refreshing after sanitization

The fresh export reproduces **every numerical value** in the existing plot CSVs.
Differences are confined to `source_sha256`, `config_sha256`, and
`submission_cohort_id` fields. All 3,139 source-file digests differ from the old
exports; 3,138 config identities differ. For all 3,138, the old exported config hash
matches the raw record's retained `original_config_sha256`; the remaining config
identity already matches.

This is explained by the documented public-release sanitization in
`docs/provenance/README.md`: portable paths changed embedded config identities and
result-file bytes while retaining measurements. No current cohort membership or
numeric aggregate changed on regeneration. Refresh the plot exports to restore
checksums and identities that trace directly to the current published raw files.
The dashboard3 `--check` alone does not detect this: it compares generated outputs
with existing exports, rather than verifying those exports against raw evidence.

## Historical material and limits

Part 2's Questions 1–8 remain a September narrative snapshot, as its README states.
Its coverage and selected October highlights are generated, but refreshing
Dashboard3 does not recompute the older chart constants. Do not interpret the
page-wide October evidence date as the date of every narrative value.

The details drawer contains 82 historical experiment entries. Refreshed metrics
that have a corresponding current performance summary matched. E250 and E251
remain historical one-iteration canary entries; they are not current performance
summaries and should not be used as repeated benchmark evidence. The catalog
correctly identifies them as canaries.

The report CSV generator belongs to an external dissertation pipeline. This audit
independently reconciled the metric values, the ratios listed above, and all goodput
means; it does not certify every historical prose claim, quality metric, or
external report transformation. Some findings in the generator are literal text,
so reproducible generation alone does not prove every sentence remains justified
when inputs change.

## Reproduction artifacts

The separate regenerated exports, executable audit, and machine-readable results
are in `build/dashboard3-audit/`. The audit uses current checked-in raw evidence
and does not overwrite `results/plot_data/`.

Commands used:

```bash
python3 dashboard3/build/build_dashboard3.py --check
PYTHONPATH=src python3 scripts/export_plot_data.py --output-dir build/dashboard3-audit/plot_data
python3 build/dashboard3-audit/audit.py
```

The machine-readable results also record the subsequent independent Student-t
check and original/public config-hash reconciliation.
