# Research Result Retention

Raw matrix result JSON files are primary research evidence and are committed to Git. Generated tables, analysis
summaries, scheduler logs, profiler traces, model checkpoints, and engine artefacts remain reproducible or too large
for normal Git and stay ignored.

## Tracked Evidence

The repository tracks these platform-qualified raw matrix-result layouts:

```text
results/raw/l40s/E123/E123-<timestamp>-r001-<uuid>.json
results/raw/h100/E121/E121-<timestamp>-r001-<uuid>.json
results/raw/l40s/SHA256SUMS
results/raw/h100/SHA256SUMS
```

The per-trial directory layout is preferred because it preserves independent repetitions and prevents later runs
from overwriting earlier evidence. Both completed and failed matrix trials should be retained: failures and OOMs are
part of the experimental record even though analysis excludes them from performance statistics.

Local and Mac development results remain ignored because they are workflow checks rather than H100/L40S evidence.

Older matrix records without top-level platform fields may be retained unchanged. They are accepted only when their
embedded hardware identifies exactly one L40S/H100 platform. Derived outputs synthesize the platform-qualified
condition and mark it `platform_specific`; ambiguous or contradictory legacy records are rejected.

## Files That Must Not Be Committed

- Slurm `.out` and `.err` files, except short manually curated excerpts in a research note.
- PyTorch/Nsight profiler traces and raw `nvidia-smi` sampling dumps.
- Hugging Face caches, model checkpoints, TensorRT-LLM engines, datasets, virtual environments, or containers.
- Generated `results/processed`, `results/analysis`, and `results/figures` outputs; rebuild these from raw evidence.
- Credentials, access tokens, private keys, environment files, or site-local Slurm profiles.

## Importing HPC Results

Copy cluster results into a staging directory, then use the importer. For example, adapt the host and remote path:

```bash
rsync -av USER@HPC_HOST:/path/to/hpc-ai-performance-engineering/results/raw/ /tmp/hpc-result-staging/
python3 -m hpc_ai_perf.cli import-result-evidence \
  --source /tmp/hpc-result-staging \
  --result-root results/raw
```

Do not use `--delete`: the local evidence directory may contain trials copied from earlier cluster runs. The importer
preserves source bytes, validates result shape and platform identity, rejects credential-like keys, refuses a
different file at an existing immutable path, and updates the platform `SHA256SUMS` manifest.

## Validation Before Every Result Commit

Run:

```bash
make validate-result-evidence
git status --short results/raw
git diff --cached --check
```

Then inspect every new JSON record and confirm:

- `experiment_id` matches its directory and `condition_id` matches `E###@l40s` or `E###@h100` for new records;
- `status`, requested/observed GPU model, GPU count, precision, framework, and strategy are correct;
- `git_commit` is present and `git_dirty` is `false` for completed matrix evidence;
- `config_sha256`, model revision, dataset revision, command, Slurm job ID, and software versions are present;
- L40S records identify L40S and PCIe and are not stored under an H100-labelled experimental condition;
- `SHA256SUMS` covers every platform result file and `make validate-result-evidence` reports no mismatch;
- no secret, private path, or unrelated personal data appears in `command`, `error`, or environment fields.

After aggregation, analysis also rejects matrix rows with a missing or inconsistent platform identity. Every generated
summary carries `platform_id` and `comparability`; hardware summaries retain condition IDs so an intentional H100–L40S
comparison starts from separate per-platform statistics.

Commit one experiment or one tightly related trial batch at a time. A useful commit subject is:

```text
results(l40s): add E123 GPT-2 training trials
```

Git provides versioned redundancy but is not the only backup. Keep the original HPC copy until the commit has been
pushed and verified on the remote repository, and retain a second archival copy for dissertation submission.

## R and JavaScript plotting datasets

`make export-plot-data PYTHON=python` creates study-specific CSV/JSON exports in
[`results/plot_data/`](plot_data/README.md), with cohort selection, provenance,
condition summaries, scaling contrasts and request distributions. The folder
structure, examples and generator are tracked; the generated data remains ignored
and reproducible from committed raw evidence. See its data dictionary before
combining trials or selecting scaling baselines.

## Public release metadata

Internal node labels and two CUDA installation paths were generalized for release.
See [the transformation record](../docs/reproducibility.md#public-metadata-transformation).
The manifests verify the published records; privately archived originals retain
their original bytes. Scientific measurements were not changed.
