# Open source release preparation

The working tree and historical source have been sanitized locally. Repository
visibility and remote refs have not been changed. The original Git repository,
result records, operational notes, and pre-cleanup files are backed up privately.

## Publication inventory

- Keep source, configurations, schemas, tests, research methodology, limitations,
  sanitized canonical evidence, and the canonical `dashboard/`.
- Keep `dashboard3/` and `results/report_data/` locally and ignored pending a
  long-term export policy. They are excluded from release artifacts.
- Obsolete `dashboard-2/`, empty requirements.txt, and unused placeholder scaffolding
  were removed. Generators create their output directories on demand.
- Date-specific reservation, priority, pending-work, and publication brainstorming
  notes were privately archived and removed throughout public-facing history.
- MIT was selected for project code; third-party models, datasets, and dependencies
  retain their own rights and access requirements.

## History and provenance

See [the provenance record](provenance/README.md), including the composed original/intermediate commit mapping,
branch-tip mapping, metadata transformations, and four pre-existing missing source
commits. Executable/configuration inputs are unchanged throughout mapped history.
Internal installation/node metadata and author email were sanitized. Local application
checkpoint refs were excluded from the public history copy.

## Validation

- 1,292 Python tests passed, including optional PyTorch paths.
- Ruff, dashboard checks, configuration validation, current evidence verification,
  refresh, freshness checks, and static-site build passed from a fresh sanitized clone.
- All 101,250 distinct historical manifest/evidence checks passed, covering 127
  affected historical manifest versions. Identified private metadata was absent
  from all reachable rewritten objects.
- The prior source cleanup verified all 3,139 current result records against the
  private backup; changes are limited to documented metadata substitutions.
- The wheel built, installed into an isolated target, and its CLI help ran outside
  the checkout during source cleanup. Local Markdown links were checked.
- Dedicated Gitleaks scans use narrow reviewed false-positive rules for scientific
  input hashes and dashboard metric arrays; no result directories are excluded.
  Repeat the scan after any changes or remote replacement.

```bash
gitleaks git --redact --log-opts="--all"
make check validate validate-result-evidence
make refresh check-dashboard-data build-dashboard-site
```

Local validation uses Python 3.13 and Node.js 24; CI declares Python 3.11 and Node.js
22 and requires its own run. GPU experiments were not rerun during repository cleanup.

## Remaining publication steps

1. Review the sanitized diff and ownership/redistribution rights.
2. Review the old/new remote branch tips and the remote history replacement. Keep
   the remote private until all intended refs are sanitized and checked. This step
   was not performed by the local rewrite.
3. Publish the sanitized source, rerun CI, and address any GitHub-specific retained
   refs or caches if necessary. Do not merge the original history back in.
4. Freeze the release tag and report permalinks using the sanitized full commit IDs.
   Cite mapped source commits for historical code and document provenance gaps.

Contributor validation is separate from credentialed dashboard publication. Only
trusted main-branch pushes or manual invocation can run dashboard deployment.

## Additional personal-document review

A second pass removed three further personal advice and handoff documents from all
public branch histories. Retained study designs use neutral language; reusable
commands use relative checkout and sibling-bundle paths. Personal home prefixes in
result commands, profiler paths, and embedded resolved configs were normalized.
See the provenance guide for original/public config hashes and composed commit IDs.
All numeric measurements and workload/runtime fingerprints were retained.
The publication-path checker covers home-prefix case variants and runs in CI.
