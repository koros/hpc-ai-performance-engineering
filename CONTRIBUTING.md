# Contributing

Use Python 3.11 or later. From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,yaml]"
make check
make validate
make validate-result-evidence
```

Dashboard tests require Node.js 22 or later. GPU experiments require the optional workload
packages, a compatible CUDA stack, and the platform acceptance gates described
in the runbooks. CPU tests do not establish GPU performance or compatibility.

Keep changes focused and describe the problem, resulting behaviour, and validation
in the pull request. Preserve experiment IDs and recorded trial provenance.
Do not change historical measurements or combine incompatible cohorts.
Use the evidence importer for new trials and include updated checksum manifests.

Do not commit credentials, local profiles, private notes, model weights, or caches.
Report generated output changes with their reproduction commands. Public discussion
should be respectful and focus on evidence and reproducibility.
