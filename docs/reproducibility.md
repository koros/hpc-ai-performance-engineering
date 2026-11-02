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

## Dashboard publication

The canonical dashboard is published directly from this repository at
<https://koros.github.io/hpc-ai-performance-engineering/>. The root opens the
catalog; Part 3 is at `dashboard/studies.html` below that URL.

`.github/workflows/dashboard-pages.yml` validates the repository, builds only the
allowlisted static files under `build/pages/`, uploads a Pages artifact and
deploys it to the `github-pages` environment. Deployment uses GitHub's job token
and OIDC; no cross-repository SSH deploy key is needed. In the repository's
Settings → Pages, the build source must be **GitHub Actions**.

The required curated report CSV snapshots are included in `results/report_data/`
so a fresh checkout can reproduce the studies. Plot exports are regenerated from
verified raw evidence before dashboard checks. The external dissertation R
pipeline is not needed to publish this validated snapshot.

The former `hpc-ai-performance-engineering-dashboard` repository is no longer
updated by this workflow. Its existing site is retained during the transition.

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
All measurements, experiment IDs, trial identifiers, runtime fingerprints,
and timestamps were retained. Platform SHA256SUMS manifests were regenerated for
the published bytes. These checksums verify the public records, not the privately
archived originals. Node labels preserve within-study grouping without publishing
internal hostnames. This is a documented metadata transformation, not a rerun.

## Report links and release gate

Cite a final release and full commit SHA. For code quotations, use GitHub's commit
permalink with line anchors; branch links and the live dashboard may change.
Record experiment IDs, platform, cohort, and trial counts alongside figures.
Archive the exact static dashboard used by the report.

The original source history has now been sanitized locally. Use the
[old-to-new commit mapping](provenance/README.md) to resolve recorded experiment
commits. Four pre-existing missing source commits are documented there. The remote
still requires a reviewed history replacement before publication; see
[release preparation](open_source_release.md).

## Portable paths and resolved configuration identities

A subsequent review removed personal advice notes and normalized personal workspace
prefixes in runbooks and historical result metadata. Repository paths are relative
to the checkout; submission bundles use `../hpc-ai-submissions/`. Run commands from
the repository root and choose local bundle locations as needed.

For affected embedded resolved configurations, `config_sha256` now hashes the public
configuration and `original_config_sha256` retains its original recorded identity.
The [configuration hash map](provenance/config-hash-map.json) links those identities.
Source-config file digests, workload/runtime fingerprints, timestamps, measurements,
experiment IDs and original recorded Git IDs are retained. Public file checksum
manifests were updated again. The original resolved configuration can be verified
against the private archive; the public embedded config verifies against its new
public hash. These are documented metadata transformations, not new trials.

Check tracked files with `python scripts/check_publication_paths.py`; add `--history`
to inspect objects reachable from local branches and tags. The check covers both
uppercase and lowercase home-directory prefixes and runs in contributor CI.
