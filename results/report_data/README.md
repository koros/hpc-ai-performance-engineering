# Curated dashboard report inputs

These twelve CSV snapshots are the report aggregates consumed or cited by
`scripts/build_dashboard_studies.py`. They were exported by the external
report pipeline on 27 October 2026. They contain numerical research summaries,
experiment IDs, settings and cohort identities; raw evidence remains under
`results/raw/`.

They are checked in so a fresh public checkout can reproduce Part 3 without
access to the external dissertation workspace. Additional local report exports
remain ignored. The dashboard refresh regenerates verified plot datasets from
raw evidence, refreshes report-based chart intervals from current trial summaries,
and recomputes goodput from request latencies and measured durations. It does
not rerun the external R report pipeline.

After new experiment evidence, refresh these curated snapshots using the report
pipeline and reconcile their means and ratios against the current plot exports
before publishing. See `docs/dashboard3-data-audit.md` for the current validation.
