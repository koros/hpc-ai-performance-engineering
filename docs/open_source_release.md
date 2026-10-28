# Open source release preparation

The cleanup branch prepares the public working tree. Repository visibility,
remote history, tags, and publication have not been changed.

## Publication inventory

- Keep source, configurations, schema, tests, research methodology, limitations,
  canonical evidence, and the canonical `dashboard/`.
- Keep `dashboard3/` and `results/report_data/` locally and ignored pending a
  long-term export policy. They are not release inputs.
- Remove obsolete `dashboard-2/`, empty requirements.txt, and unused placeholder
  scaffolding. Generated result directories are created on demand.
- Archive date-specific reservation, weekend priority, pending-work, and publication
  brainstorming notes privately outside the repository. Preserve reusable runbooks.
- Keep the original pre-cleanup files and evidence in a private external backup.

## Remaining release gates

1. Older commits contain original node names, infrastructure paths, archived notes,
   and author email metadata. Do not make the existing history public without a
   deliberate history-publication decision. A clean public snapshot is an option;
   preserving history requires a scoped rewrite and a repeat audit of all public
   refs. Keep the original repository privately either way. Do this before fixing
   report permalinks; rewritten commit hashes will change.
2. Review ownership and third-party redistribution rights before publication.
   MIT was selected by the maintainer for project code; it grants no rights to
   third-party models, datasets, or dependencies.
3. Review the final diff, run documented checks from a clean export, and repeat a
   dedicated secret scan over the final public tree and all intended public refs.
   Pattern scans alone cannot certify that no sensitive content exists.
4. Create the final release tag and update report citations only after these gates.

Validation and deployment are separate workflows. Dashboard deployment requires
repository credentials and runs only from trusted main-branch pushes or manual
invocation. Contributor validation does not require deployment secrets.

## Audit results

Gitleaks scanned 561 reachable commits (about 84.7 MB of changed content). Its
123 initial findings were reviewed: 117 were SHA-256 fingerprints of tokenized
research inputs and six were dashboard metric-name arrays. Narrow line-pattern
allowlists in `.gitleaks.toml` document those false positives; no directories are
excluded. A separate scan of 6,650 reachable blobs confirmed that historical
personal paths and internal node metadata remain. Secret scanning does not
remove this history-publication gate.

To repeat the checks with Gitleaks installed:

```bash
gitleaks git --redact --log-opts="--all"
gitleaks dir build/public-release --redact
```

A local source export under `build/public-release/` omits Git history, local
dashboards, report exports, caches, and private archives. It is a review artifact,
not a published release. Rebuild it after further source changes.

## Local validation completed

- 1,291 Python tests passed, including optional PyTorch paths.
- Ruff, dashboard tests, configuration validation, and raw-evidence checks passed.
- Refresh, dashboard freshness check, and static-site build passed from a source
  export without Git history or local report/dashboard folders.
- All 3,139 result files were compared against the private backup: differences
  are limited to the documented metadata substitutions, including node names
  inside serialized rank placement.
- The wheel built, installed into an isolated target directory, and its CLI help
  ran outside the checkout. Local Markdown destinations were checked.
- Gitleaks reported no remaining findings in the source export or all reachable
  history with the reviewed narrow false-positive rules. GPU execution was not
  rerun as part of this cleanup.

Validation used Python 3.13 and Node.js 24 locally; CI declares Python 3.11 and
Node.js 22 and still requires its own run.
