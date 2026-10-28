# Reproduction and public evidence

## CPU validation

Use Python 3.11 or later and Node.js 22 or later for dashboard tests:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,yaml]"
make check
make validate
make validate-result-evidence
make refresh
make check-dashboard-data
make build-dashboard-site
```

The static site is written under `build/pages/`. Analysis tables and figures are
generated outputs; their directories are created by the generators. Plot exports
can be regenerated with `make export-plot-data`. See
[the data dictionary](../results/plot_data/DATA_DICTIONARY.md) before pooling trials.

## GPU reproduction

See [platform execution](platform_execution_runbook.md),
[measurement protocol](measurement_protocol.md), and
[workload capabilities](workload_catalog.md). Cluster profiles are examples of the
measured environment. Supply your own local account, paths, modules, partitions,
and reservations; local profile overrides are ignored by Git. Setup commands may
install dependencies or submit jobs and should be reviewed before use.

Model weights and datasets are not bundled. Their licenses, publisher revisions,
access conditions, and runtime requirements remain independent of this project's
MIT code license. Historical records retain their recorded software versions;
installing current packages does not reproduce an old environment automatically.

## Public metadata transformation

During release preparation on 28 October 2026, internal node names in 3,139
canonical result records (including serialized rank-placement metadata) were replaced consistently with `research-node-a` and
`research-node-b`. Two E005 records also had internal CUDA installation paths
replaced with `/opt/site/cuda-12.4.1`, retaining the `bin/nvcc` suffix where present.
All measurements, experiment IDs, trial identifiers, runtime/config fingerprints,
and timestamps were retained. Platform SHA256SUMS manifests were regenerated for
the published bytes. These checksums verify the public records, not the privately
archived originals. Node labels preserve within-study grouping without publishing
internal hostnames. This is a documented metadata transformation, not a rerun.

## Report links and release gate

Cite a final release and full commit SHA. For code quotations, use GitHub's commit
permalink with line anchors; branch links and the live dashboard may change.
Record experiment IDs, platform, cohort, and trial counts alongside figures.
Archive the exact static dashboard used by the report.

Before changing repository visibility, complete the history review described in
[release preparation](open_source_release.md). Current-tree redaction does not
remove earlier versions from Git history.
