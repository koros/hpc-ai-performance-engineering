# L40S-First Platform Execution Runbook

## Objective

Collect useful L40S evidence while the H100 node is unavailable, without losing the ability to run matched H100
conditions later. Logical experiment configs stay unchanged; the platform policy creates generated execution configs
and keeps raw evidence physically and semantically separate.

## Invariants

- `E###@l40s` and `E###@h100` are different conditions even when they share a logical `E###` row.
- Accepted L40S evidence goes only under `results/raw/l40s/E###/`; accepted H100 evidence goes only under
  `results/raw/h100/E###/`.
- Generated Slurm jobs write trial JSON into ignored staging paths under `build/platform_results/<platform>/wave_<wave>/`;
  import accepted records into `results/raw/` after each wave.
- A wave advances only after its required trials are valid in the platform tracker.
- `exact` results preserve workload shape. `platform_specific` results describe platform behaviour and are not pooled.
- A capacity-driven workload change must be registered as `capacity_adjusted` before execution.
- Failed and OOM runs remain immutable evidence; they do not count as valid performance trials.

## 1. Repository and Cluster Preflight

From a clean checkout on the HPC login node:

```bash
source slurm/modules_for_hpc_ai
python3 -m pip install \
  --constraint slurm/constraints/lovelace_py311_cuda124.txt \
  --upgrade-strategy only-if-needed \
  -e ".[dev,yaml,workloads,distributed]"
python3 -m hpc_ai_perf.cli validate --skip-results --research-ready
python3 -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
sinfo -p gpu_l40s,gpu_h100
```

Do not use `--upgrade-strategy eager` on Lovelace; it can mix PyPI Torch/Numpy wheels with the Spack CUDA/PyTorch
runtime and invalidate both DeepSpeed and vLLM checks.

The current execution matrix deliberately defers E006 and all other vLLM-serving rows. The PyTorch CUDA runtime is
healthy, but the site has neither a compatible vLLM package nor a usable CUDA compiler toolchain for a supported build.
Do not install a generic PyPI vLLM wheel into this environment. Re-enable those rows only after an administrator
provides a compatible vLLM module, container, or build environment and its dedicated preflight passes.

On an L40S allocation, capture the actual device and topology before accepting results:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi topo -m
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

Stop if the observed device is not L40S, CUDA is unavailable, or the generated script requests another partition.

Before rerunning wave 0 after an environment failure, run the active-node checks inside matching allocations:

```bash
srun -p gpu_l40s --gres=gpu:2 --cpus-per-task=8 --time=00:10:00 bash -lc '
cd ./
source slurm/modules_for_hpc_ai
source .venv/bin/activate
python3 -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --expected-torch-version 2.8.0 --expected-torch-cuda 12.4 --expected-gpus 2 --require-deepspeed
'

```

The DeepSpeed check may warn when `nvcc` is unavailable, but it must still pass `deepspeed_import`. If DeepSpeed 0.17.x
is installed without a full CUDA toolkit, upgrade to the project-supported DeepSpeed 0.18.9+ line or load a toolkit whose
`CUDA_HOME/bin/nvcc` exists before running E005. The current wave 0 scope is E001-E005; do not run a vLLM check or E006
until the separate serving environment has been supplied.

## 2. Preserve and Import L40S Trials

Do not delete a failed or provenance-rejected staging directory. Move it to a clearly named quarantine/archive
directory and retain its Slurm logs for diagnosis. New certified runs use an immutable submission bundle, whose
`results/SUBMISSION_MANIFEST.json` travels with the JSON records and proves the exact source/configuration snapshot.

Import from the completed bundle, including while other jobs from the same bundle remain queued:

```bash
export SUBMISSION_BUNDLE_DIR=../hpc-ai-submissions/<bundle-id>
make import-platform-wave-bundle SUBMISSION_BUNDLE_DIR="$SUBMISSION_BUNDLE_DIR"
make validate-result-evidence
make track-l40s-progress
```

The importer preserves original JSON bytes, verifies the adjacent submission manifest for bundle results, infers one
unambiguous platform, rejects credential-like fields and invalid records, refuses conflicting overwrites, and refreshes
`results/raw/l40s/SHA256SUMS`. An empty `results/raw/h100/` directory is harmless; it does not require an H100
checksum manifest until it contains evidence.

Inspect and commit one experiment or tightly related batch at a time:

```bash
git status --short results/raw/l40s
git diff -- results/raw/l40s
git add results/raw/l40s/E123 results/raw/l40s/SHA256SUMS
git commit -m "results(l40s): add E123 trials"
```

Keep the HPC originals until the commits are pushed and independently backed up.

## 3. L40S Collection Waves

| Wave | Conditions | Trials | Release gate |
| --- | --- | ---: | --- |
| `0` | `E001-E005` core environment validation | 1 | All five Complete; observed L40S/CUDA/distributed/DeepSpeed checks agree. |
| `1` | `E008-E015`, `E018-E031` baselines and characterisation | 3 | Core rows Complete; variance and provenance reviewed. |
| `2` | `E123`, `E124`, `E127`, `E128` direct L40S comparison anchors | 3 | All matched L40S anchors Complete. |
| `3` | `E032-E043`, `E047-E062` scaling and strategies | 3 | 1-GPU reference valid; 2/4-GPU pilot stable. |
| `4` | Supported memory, precision, and data movement rows | 3 | No silent OOM-driven shape changes; precision backend observed. |
| `4b` | `E113-E120` communication | 3 | Interpret as NCCL over installed PCIe topology; no NVLink claims. |
| `5` | Supported PyTorch inference rows | 3 | PyTorch engine, request shape, concurrency, and precision verified; vLLM rows stay deferred. |
| `optional` | `E016-E017`, `E044-E046` | 3 | Run only if memory and queue budget remain. |

E006, E007, all vLLM-serving rows, and backend/adapter rows listed as Deferred are not rendered by platform batch commands.

## 4. Create, Review, and Submit an Immutable Wave Bundle

From a login node, load the exact module stack and virtual environment which the generated job will activate. Commit
all changes to `src/`, `configs/`, `platforms/`, `slurm/`, and `pyproject.toml` first. Result evidence, reports, and
ignored build output may remain present: they are deliberately excluded from the source-snapshot cleanliness check.

```bash
cd ./
source slurm/modules_for_hpc_ai
source .venv/bin/activate

make create-platform-wave-bundle \
  PLATFORM=l40s PLATFORM_WAVE=0 \
  SUBMISSION_BUNDLE_ID=l40s-wave-0-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
```

Record the printed path, then initialize and submit from that immutable bundle:

```bash
export SUBMISSION_BUNDLE_DIR=../hpc-ai-submissions/<bundle-id>
make plan-platform-wave-bundle PLATFORM=l40s PLATFORM_WAVE=0
make init-platform-wave-bundle-status PLATFORM=l40s PLATFORM_WAVE=0
make dry-run-submit-platform-wave-bundle PLATFORM=l40s PLATFORM_WAVE=0
make submit-platform-wave-bundle PLATFORM=l40s PLATFORM_WAVE=0
```

The bundle contains `repo/` (detached source snapshot), `configs/` (frozen resolved configs), `scripts/`, `logs/`,
`results/`, `SUBMISSION_MANIFEST.json`, and `SCRIPT_SHA256SUMS`. Every task validates its source commit, clean
snapshot, config hash, manifest hash, and runtime dependency contract before invoking the workload. Its logs and JSON
never modify the main repository checkout.

You may import and commit completed trial evidence while later jobs are queued. Do not alter an existing bundle or add
new scripts to it; create another bundle for a changed commit, environment, platform, wave, or experiment selection.
Do not advance based only on Slurm `COMPLETED`; the platform tracker must show valid trial evidence.

## 5. H100 Recovery Path

When `gpu_h100` returns, verify the node and start its independent wave 0:

```bash
python3 -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_h100.json
source slurm/modules_for_hpc_ai
source .venv/bin/activate
make create-platform-wave-bundle \
  PLATFORM=h100 PLATFORM_WAVE=0 \
  SUBMISSION_BUNDLE_ID=h100-wave-0-$(git rev-parse --short HEAD)-$(date -u +%Y%m%dT%H%M%SZ)
make track-h100-progress
```

Use the same bundle process with `PLATFORM=h100` and a separate `SUBMISSION_BUNDLE_DIR`. H100 staged JSON is accepted
under its own evidence root, never changing L40S status. Only compare rows whose workload, revisions, seed,
useful-work definition, and `comparability` contract match. The hardware-comparison anchors are `E121/E123`,
`E122/E124`, `E125/E127`, and `E126/E128`.

## 6. Recovery Rules

- Failed trial: retain it, diagnose, and rerun the same repetition. The tracker counts only the latest record for that
  repetition and does not inflate the sample size.
- Invalid result: fix the execution/provenance issue and rerun; never hand-edit primary evidence.
- Conflicting import filename: keep both originals outside Git, identify why run IDs collided, and resolve the workflow
  before importing. The importer will not overwrite either side.
- H100 still down: continue only L40S planned waves. Record H100 rows as Ready/Deferred, not Failed or Complete.
- L40S memory mismatch: stop the exact condition and register a capacity-adjusted condition before changing batch or
  accumulation settings.
