# Research Result Retention

Raw matrix result JSON files are primary research evidence and are committed to Git. Generated tables, analysis
summaries, scheduler logs, profiler traces, model checkpoints, and engine artefacts remain reproducible or too large
for normal Git and stay ignored.

## Tracked Evidence

The repository tracks only these raw matrix-result layouts:

```text
results/raw/E123.json
results/raw/E123/E123-<timestamp>-r001-<uuid>.json
```

The per-trial directory layout is preferred because it preserves independent repetitions and prevents later runs
from overwriting earlier evidence. Both completed and failed matrix trials should be retained: failures and OOMs are
part of the experimental record even though analysis excludes them from performance statistics.

Local and Mac development results remain ignored because they are workflow checks rather than H100/L40S evidence.

## Files That Must Not Be Committed

- Slurm `.out` and `.err` files, except short manually curated excerpts in a research note.
- PyTorch/Nsight profiler traces and raw `nvidia-smi` sampling dumps.
- Hugging Face caches, model checkpoints, TensorRT-LLM engines, datasets, virtual environments, or containers.
- Generated `results/processed`, `results/analysis`, and `results/figures` outputs; rebuild these from raw evidence.
- Credentials, access tokens, private keys, environment files, or site-local Slurm profiles.

## Importing HPC Results

Copy the immutable JSON trial directories from the HPC repository into the same local path. For example, adapt the
host and remote repository path before running:

```bash
rsync -av --include='E[0-9][0-9][0-9]/***' --exclude='*' \
  USER@HPC_HOST:/path/to/hpc-ai-performance-engineering/results/raw/ results/raw/
```

Do not use `--delete`: the local evidence directory may contain trials copied from earlier cluster runs.

## Validation Before Every Result Commit

Run:

```bash
make validate-result-evidence
git status --short results/raw
git diff --cached --check
```

Then inspect every new JSON record and confirm:

- `experiment_id` matches its directory and intended config;
- `status`, requested/observed GPU model, GPU count, precision, framework, and strategy are correct;
- `git_commit` is present and `git_dirty` is `false` for completed matrix evidence;
- `config_sha256`, model revision, dataset revision, command, Slurm job ID, and software versions are present;
- L40S records identify L40S and PCIe and are not stored under an H100-labelled experimental condition;
- no secret, private path, or unrelated personal data appears in `command`, `error`, or environment fields.

Commit one experiment or one tightly related trial batch at a time. A useful commit subject is:

```text
results(l40s): add E123 GPT-2 training trials
```

Git provides versioned redundancy but is not the only backup. Keep the original HPC copy until the commit has been
pushed and verified on the remote repository, and retain a second archival copy for dissertation submission.
