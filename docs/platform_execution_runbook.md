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

The current execution matrix deliberately defers E006 and all other vLLM-serving rows until their isolated environment
is built. The verified Lovelace helper now provides a CUDA 12.4.1 compiler for the pinned source build, but vLLM is not
part of the shared environment. Do not install a generic PyPI vLLM wheel into this environment. Follow the
[vLLM environment runbook](vllm_environment_runbook.md) to build it privately and run the E006 serving acceptance gate.
Re-enable these rows only after the vLLM build and dedicated preflight pass.

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

The helper must expose `nvcc` and set `CUDA_HOME` to the matching compiler root before running E005. Run
`make check-cuda-toolchain PLATFORM=l40s EXPECTED_GPUS=2` in this allocation to verify the complete toolchain. The
current wave 0 scope is E001, E002, E004, and E005; E003 requires four GPUs and is deferred on the active L40S node.
Do not run a vLLM check or E006 until the separate serving environment passes its dedicated preflight.

## 2. Preserve and Collect L40S Trials

Do not delete a failed or provenance-rejected staging directory. Move it to a clearly named quarantine/archive
directory and retain its Slurm logs for diagnosis. New certified runs use an immutable submission bundle, whose
`results/SUBMISSION_MANIFEST.json` travels with the JSON records and proves the exact source/configuration snapshot.

The standard workflow stores the current bundle location in ignored `build/platform_waves/`; do not copy result files or
export a bundle-path variable. When every required trial is complete, collect and stage evidence with:

```bash
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
git diff --cached --stat
git commit -m "data: add completed L40S wave 0 trials"
```

Collection preserves original JSON bytes, verifies the adjacent submission manifest, infers one unambiguous platform,
rejects credential-like fields and invalid records, refreshes `results/raw/l40s/SHA256SUMS`, and updates the ignored
tracker. It refuses an incomplete current-policy wave rather than asking the user to select result folders manually.
If a failed row is subsequently deferred with a documented platform reason, collection combines the valid trials from
earlier attempts and no longer requires that deferred row. An empty `results/raw/h100/` directory is harmless; it does
not require an H100 checksum manifest until it contains evidence.

Keep the HPC originals until the commits are pushed and independently backed up.

## 3. L40S Collection Waves

| Wave | Conditions | Trials | Release gate |
| --- | --- | ---: | --- |
| `0` | `E001`, `E002`, `E004`, `E005` core environment validation | 1 | All four Complete; observed L40S/CUDA/distributed/DeepSpeed checks agree. |
| `1` | `E008-E015`, `E018-E031` baselines and characterisation | 3 | Core rows Complete; variance and provenance reviewed. |
| `2` | `E123`, `E124`, `E127`, `E128` direct L40S comparison anchors | 3 | All matched L40S anchors Complete. |
| `3` | L40S 1/2-GPU scaling and strategy conditions, excluding E032/E033 | 3 | Supported 1/2-GPU rows are valid. E032 and E033 are deferred after exact-shape L40S OOMs; four-GPU conditions are also deferred. |
| `4` | Supported memory, precision, and data movement rows | 3 | No silent OOM-driven shape changes; precision backend observed. |
| `4b` | `E113`, `E114`, `E117`, `E118` communication | 3 | Interpret as two-GPU NCCL over installed PCIe topology; no NVLink claims. |
| `5` | Supported PyTorch inference rows | 3 | PyTorch engine, request shape, concurrency, and precision verified; vLLM rows stay deferred. |
| `optional` | `E016-E017`, `E044-E045` | 3 | Run only if memory and queue budget remain. |

DeepSpeed's own optional NVTX annotations are disabled by default for ZeRO rows. The installed Lovelace NVTX binding
does not accept the keyword arguments used by DeepSpeed 0.19.6, while the project NCCL and `nvidia-smi` collectors
remain active. Set `HPC_AI_ENABLE_DEEPSPEED_NVTX=1` only after verifying a compatible NVTX binding for a dedicated
Nsight study. The DeepSpeed `bf16`/`fp16` configuration is the sole mixed-precision controller for ZeRO rows; the
runner does not nest PyTorch native AMP around the DeepSpeed engine.

E006, E007, all vLLM-serving rows, E032/E033, four-GPU L40S rows, and backend/adapter rows listed as Deferred are not
rendered by platform batch commands. E032 and E033 exhausted the exact GPT-2 strong-scaling shapes on the observed
L40S capacity; future reduced-batch studies must be registered as distinct capacity-adjusted conditions. The active
L40S node exposes two GPUs, so the matrix preserves only supported one- and two-GPU conditions in runnable waves.

## 4. Submit, Monitor, and Recover an Immutable Wave

From a login node, load the exact module stack and virtual environment which the generated job will activate. Commit
all changes to `src/`, `configs/`, `platforms/`, `slurm/`, and `pyproject.toml` first. Result evidence, reports, and
ignored build output may remain present: they are deliberately excluded from the source-snapshot cleanliness check.

```bash
cd ./
source slurm/modules_for_hpc_ai
make check-slurm-profile SLURM_PROFILE=slurm/profiles/cluster_l40s.json
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

The command creates the bundle, writes its Slurm status record and ignored run-state file, then submits every script.
Generated scripts activate the shared environment themselves. Monitor it with:

```bash
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

The bundle contains `repo/` (detached source snapshot), `configs/` (frozen resolved configs), `scripts/`, `logs/`,
`results/`, `SUBMISSION_MANIFEST.json`, and `SCRIPT_SHA256SUMS`. Every task validates its source commit, clean
snapshot, config hash, manifest hash, and stable runtime identities (interpreter version, package versions, and Torch CUDA
build) before invoking the workload. The project setup script itself is snapshot-hashed; inherited module-list text is
not compared because Slurm and Lmod may represent an equivalent reloaded stack differently. Its logs and JSON never
modify the main repository checkout.

Do not alter an existing bundle or add new scripts to it. Do not advance based only on Slurm `COMPLETED`:
`collect-platform-wave` requires valid trial evidence for the whole wave. If the status report lists incomplete
experiments, submit only those from a new immutable bundle:

```bash
make retry-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

## 5. H100 Recovery Path

When `gpu_h100` returns, verify the node and start its independent wave 0:

```bash
python3 -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_h100.json
source slurm/modules_for_hpc_ai
source .venv/bin/activate
make submit-platform-wave PLATFORM=h100 PLATFORM_WAVE=0
make track-h100-progress
```

Use the same `status-platform-wave`, `collect-platform-wave`, and `stage-platform-wave` commands with
`PLATFORM=h100`. H100 staged JSON is accepted
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
