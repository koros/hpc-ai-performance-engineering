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
python3 -m pip install -e ".[dev,yaml,workloads,distributed]"
python3 -m hpc_ai_perf.cli validate --skip-results --research-ready
python3 -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
sinfo -p gpu_l40s,gpu_h100
```

Before E006 or any later vLLM wave, install vLLM separately for the exact active CUDA/PyTorch/driver stack. If PyTorch
reports CUDA 12.x but `import vllm` fails looking for `libcudart.so.13`, the installed vLLM wheel targets a different
CUDA runtime and must be replaced before the result can be accepted.

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
python3 -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --expected-gpus 2 --require-deepspeed
'

srun -p gpu_l40s --gres=gpu:1 --cpus-per-task=8 --time=00:10:00 bash -lc '
cd ./
source slurm/modules_for_hpc_ai
source .venv/bin/activate
python3 -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --expected-gpus 1 --require-vllm
'
```

The DeepSpeed check may warn when `nvcc` is unavailable, but it must still pass `deepspeed_import`. If DeepSpeed 0.17.x
is installed without a full CUDA toolkit, upgrade to the project-supported DeepSpeed 0.18.9+ line or load a toolkit whose
`CUDA_HOME/bin/nvcc` exists before running E005. The vLLM check must pass before E006 can become accepted evidence.

## 2. Import L40S Trials Already Run

Copy cluster JSON into a staging directory. Do not copy directly over committed evidence and do not use `--delete`.
For newly rendered platform waves, the default source is the ignored staging directory:

```bash
python3 -m hpc_ai_perf.cli import-result-evidence \
  --source build/platform_results/l40s/wave_0 \
  --result-root results/raw
```

If results were produced on another checkout or copied back from the cluster, import from that copied staging path:

```bash
rsync -av USER@HPC_HOST:/path/to/project/build/platform_results/l40s/ /tmp/hpc-result-staging/l40s/
python3 -m hpc_ai_perf.cli import-result-evidence \
  --source /tmp/hpc-result-staging \
  --result-root results/raw
make validate-result-evidence
make track-l40s-progress
```

The importer preserves original JSON bytes, infers one unambiguous platform, rejects credential-like fields and
invalid records, refuses conflicting overwrites, and refreshes `results/raw/l40s/SHA256SUMS`. If an older result lacks
top-level `condition_id`, its embedded L40S hardware metadata still routes it to the L40S tracker; do not edit the
historical record merely to upgrade its schema.

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
| `0` | `E001-E006` environment validation | 1 | All six Complete; observed L40S/CUDA/distributed/backend checks agree. |
| `1` | `E008-E015`, `E018-E031` baselines and characterisation | 3 | Core rows Complete; variance and provenance reviewed. |
| `2` | `E123`, `E124`, `E127`, `E128` direct L40S comparison anchors | 3 | All matched L40S anchors Complete. |
| `3` | `E032-E043`, `E047-E062` scaling and strategies | 3 | 1-GPU reference valid; 2/4-GPU pilot stable. |
| `4` | Supported memory, precision, and data movement rows | 3 | No silent OOM-driven shape changes; precision backend observed. |
| `4b` | `E113-E120` communication | 3 | Interpret as NCCL over installed PCIe topology; no NVLink claims. |
| `5` | Supported inference rows | 3 | Engine, request shape, concurrency, and precision verified. |
| `optional` | `E016-E017`, `E044-E046` | 3 | Run only if memory and queue budget remain. |

`E007@l40s` and backend/adapter rows listed as Deferred are not rendered by platform batch commands.

## 4. Render, Review, and Submit One Wave

```bash
make render-l40s-wave PLATFORM_WAVE=0
make plan-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
make init-platform-wave-status PLATFORM=l40s PLATFORM_WAVE=0
make dry-run-submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

Review every generated `#SBATCH` directive and resolved config. Then submit from the repository root on the cluster:

```bash
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

Generated jobs write result JSON to `build/platform_results/l40s/wave_0/` by default and Slurm logs to
`slurm/generated/l40s/wave_0/logs/`. After jobs finish, import their JSON evidence, verify checksums, and rebuild the
tracker:

```bash
make import-result-evidence SOURCE=build/platform_results/l40s/wave_0
make validate-result-evidence
make track-l40s-progress
```

Do not advance based only on Slurm `COMPLETED`; the platform tracker must show valid trial evidence.

## 5. H100 Recovery Path

When `gpu_h100` returns, verify the node and start its independent wave 0:

```bash
python3 -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_h100.json
make render-h100-wave PLATFORM_WAVE=0
make track-h100-progress
```

Use the same wave process with `PLATFORM=h100`. H100 staged JSON is written under `build/platform_results/h100/` and
accepted evidence is imported under its own root, never changing L40S status. Only compare rows whose workload,
revisions, seed, useful-work definition, and `comparability` contract match. The hardware-comparison anchors are
`E121/E123`, `E122/E124`, `E125/E127`, and `E126/E128`.

## 6. Recovery Rules

- Failed trial: retain it, diagnose, and rerun the same repetition. The tracker counts only the latest record for that
  repetition and does not inflate the sample size.
- Invalid result: fix the execution/provenance issue and rerun; never hand-edit primary evidence.
- Conflicting import filename: keep both originals outside Git, identify why run IDs collided, and resolve the workflow
  before importing. The importer will not overwrite either side.
- H100 still down: continue only L40S planned waves. Record H100 rows as Ready/Deferred, not Failed or Complete.
- L40S memory mismatch: stop the exact condition and register a capacity-adjusted condition before changing batch or
  accumulation settings.
