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

The L40S and H100 vLLM acceptance gates have passed using private PyTorch 2.8.0/CUDA 12.4 with Gloo/NCCL and vLLM
0.10.2. Do not install that stack into the shared `.venv`. Use `make submit-vllm-pilot PLATFORM=<platform>` for E006
and `make submit-vllm-wave PLATFORM=<platform> PLATFORM_WAVE=<wave>` for dedicated performance waves; those targets
bind each platform's immutable bundle to the private environment.

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
current wave 0 scope remains E001, E002, E004, and E005. Slurm job 1773321 separately verified four concurrent L40S
GPUs on the eight-GPU node, so E003 now runs through `4gpu-pilot` without reopening completed wave 0 history. Run
E006 only through `make submit-vllm-pilot`; its separate serving environment has passed the L40S preflight.

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
| `3` | L40S one-/two-GPU scaling and strategy conditions, excluding E032/E033 | 3 | Supported one-/two-GPU rows are valid. E032 and E033 remain deferred after exact-shape L40S OOMs; four-GPU rows use `4gpu-core`. |
| `4` | Supported exact memory, precision, and data movement rows, excluding E067/E068 | 3 | No silent OOM-driven shape changes; precision backend observed. E067/E068 remain H100-exact rows and L40S OOM deferrals. |
| `training-fp8` | `E231`/`E232` Transformer Engine BF16 controls and `E086`/`E093` FP8 treatments | 3 | Run only through `make submit-fp8-training-wave` after both real-model acceptance checks pass at the current commit. Compare each treatment only with its named backend-matched control. |
| `4c` | `E201-E202`, `E205` capacity-adjusted GPT-2 L40S micro-batch probes | 3 | Interpret only as L40S capacity-boundary evidence; E203/E204/E206/E207 are confirmed OOM upper-bound probes and must not be pooled with exact E067/E068. |
| `4b` | `E113`, `E114`, `E117`, `E118` communication | 3 | Interpret as two-GPU NCCL over installed PCIe topology; no NVLink claims. |
| `5` | Supported PyTorch inference rows | 3 | PyTorch engine, request shape, concurrency, and precision verified. |
| `vllm-pilot` | `E006` | 1 | Canonical import, engine-start, and served-request evidence in the accepted private environment. |
| `vllm-framework` | `E130`, `E178` | 3 | One-GPU framework baselines pass before larger serving sweeps. |
| `vllm-fp8` | `E151`, `E199` | 3 | Online FP8 W8A8 results are compared with same-platform BF16 vLLM controls E130/E178. |
| `vllm-batching` | `E132-E137`, `E180-E185` | 3 | Batch-size sweep completes with request and generated-token metrics. |
| `vllm-concurrency` | `E138-E142`, `E186-E190` | 3 | Concurrency sweep completes with latency and throughput metrics. |
| `vllm-prompt` | `E143-E146`, `E191-E194` | 3 | Prompt-length sweep completes with TTFT and generation metrics. |
| `vllm-long-context` | `E247-E249` | 3 | Separate Qwen BF16 4,096/6,144/8,192-token one-request prefill/latency curve; run only through its maximum-shape acceptance gate. |
| `vllm-tp2` | `E147`, `E195` | 3 | Two-GPU tensor-parallel rows run on both visible L40S devices. |
| `4gpu-pilot` | `E003` | 1 | Four-rank NCCL initialization and all-reduce pass before any four-GPU performance wave. |
| `4gpu-core` | `E034`, `E037`, `E040`, `E043`, `E046`, `E048`, `E050`, `E056`, `E058`, `E172` | 3 | Exact four-GPU scaling, strategy, and PyTorch classification inference conditions complete after E003. |
| `4gpu-communication` | `E115-E116`, `E119-E120` | 3 | Interpret as four-GPU NCCL over the installed L40S PCIe topology. |
| `vllm-tp4` | `E148`, `E196` | 3 | Submit with `make submit-vllm-wave` only after a four-GPU TP=4 canary succeeds. |
| `optional` | `E016-E017`, `E045` | 3 | E044 is excluded after three deterministic exact-shape L40S OOMs; retain its failed bundle for the report. |
| `optional-capacity` | `E208-E209` checkpointed TinyLlama strong scaling | 3 | Both rows must complete under the same checkpointed protocol before calculating capacity-adjusted speedup. Do not pair E208 with non-checkpointed E045. |
| `telemetry-revalidation` | `E210-E213` UUID-bound training, inference, and communication anchors | 3 | Run before final power/utilisation/energy interpretation. All records must identify the allocated GPU mapping and pass the telemetry audit. |
| `energy-revalidation` | `E214` measured-region two-GPU communication anchor | 3 | Use for communication average power, joules, and tokens/kWh after schema-v3 boundary validation. |
| `hardware-energy-revalidation` | `E235-E238` L40S hardware-comparison energy anchors | 3 | Supplies verified measured-region energy for E123/E124/E127/E128 while preserving the original rows' performance metrics. |
| `8gpu-pilot` | `E215` eight-rank NCCL and PCIe topology gate | 1 | Verify all eight allocated L40S GPUs, the rank mapping, NCCL all-reduce, and the captured PCIe topology before full-node performance work. |
| `8gpu-scaling` | `E216-E220` GPT-2, DistilBERT, and TinyLlama scaling | 3 | L40S-only full-node extension. Preserve the stated strong/weak scaling protocol; do not pool it with a differently sized H100 run. |
| `8gpu-strategy` | `E221-E224` GPT-2 and DistilBERT DDP/FSDP | 3 | Compare strategies only within the same eight-GPU L40S allocation class. |
| `8gpu-communication` | `E225-E228` GPT-2 and DistilBERT DDP/FSDP communication | 3 | Interpret as eight-GPU PCIe/NCCL behaviour using E215's topology capture, never as NVLink evidence. |
| `vllm-tp8-canary` | `E230` TinyLlama TP=8 engine-start/request gate | 1 | Use the private vLLM environment; it must pass before E229. |
| `vllm-tp8` | `E229` TinyLlama TP=8 serving | 3 | L40S-only extension. TinyLlama has 32 attention heads and is divisible by TP=8; GPT-2 is not. |

If the original optional bundle was submitted before E044 was deferred, pull the updated policy and rerun
`status-platform-wave`. It will report 9/9 current-policy trials and identify E044 as a policy-deferred historical
condition. Collect that bundle normally; do not submit another exact E044 retry.

DeepSpeed's own optional NVTX annotations are disabled by default for ZeRO rows. The installed Lovelace NVTX binding
does not accept the keyword arguments used by DeepSpeed 0.19.6, while the project NCCL and `nvidia-smi` collectors
remain active. Set `HPC_AI_ENABLE_DEEPSPEED_NVTX=1` only after verifying a compatible NVTX binding for a dedicated
Nsight study. The DeepSpeed `bf16`/`fp16` configuration is the sole mixed-precision controller for ZeRO rows; the
runner does not nest PyTorch native AMP around the DeepSpeed engine.

E007, E032/E033/E067/E068 exact L40S rows, and backend/adapter rows listed as Deferred are
not rendered by platform batch commands. L40S E006 and supported GPT-2/TinyLlama vLLM rows are rendered only by their
dedicated private-environment waves, including `vllm-tp4`. E032 and E033 exhausted the exact GPT-2 strong-scaling
shapes on the observed L40S capacity. E067 and E068 exhausted the exact GPT-2 micro-batch 32/64 shapes. Reduced-batch
L40S studies are registered separately as capacity-adjusted conditions. E201/E202 succeeded at micro-batch 20/24,
E205 succeeded at micro-batch 25, and E206/E207/E203/E204 OOMed at 26/27/28/30. The current L40S boundary is therefore
micro-batch 25 as the largest passing integer probe. The L40S node exposes eight GPUs, and a four-GPU allocation has
been verified. `8gpu-pilot` is the canonical full-node allocation and topology gate; only its accepted evidence releases
the eight-GPU supplemental waves. Those L40S-only rows do not alter the original H100 source configurations.

E247-E249 are a supplemental Qwen BF16 long-context continuation, not additional TinyLlama/GPT-2 points. They need
the private vLLM environment and the E249 8,192-token target-platform acceptance gate before `vllm-long-context` is
submitted. Use `make submit-long-context-wave`, not the generic vLLM wrapper; see the
[long-context runbook](long_context_serving_runbook.md). They are one-GPU jobs and are not released by any TP gate.

The exact one-GPU TinyLlama strong-scaling row E044 also exceeded L40S memory in all three trials. Each run had
43.30 GiB allocated by PyTorch with 592.25 MiB free before a further 1000 MiB allocation failed. E044 is therefore
preserved unchanged as an exact H100 condition and deferred only on L40S. The supplemental E208/E209 pair applies
activation checkpointing consistently to the original one-/two-GPU workload shapes and is labelled
`capacity_adjusted`; it must not be pooled with exact E044/E045 evidence.

### L40S hardware-energy revalidation sequence

The original E123/E124/E127/E128 L40S hardware-comparison records retain valid throughput, latency, and memory
evidence, but their legacy full-window telemetry is not accepted as canonical measured-region energy. Run the four
new immutable anchors instead of creating duplicate repetitions under the original experiment IDs:

```bash
source slurm/modules_for_hpc_ai
source .venv/bin/activate

make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
make verify-hardware-energy-revalidation PLATFORM=l40s
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
git commit -m "data: add L40S hardware energy revalidation trials"
```

The verifier requires three sufficient schema-v3 measured-region records for every anchor, positive finite energy
and power, at least two samples per measured region, and an allocation-bound GPU mapping. After the gate passes,
`make refresh` attributes only the validated E235-E238 energy evidence to E123/E124/E127/E128 in
`hardware_summary.csv`; it does not substitute anchor throughput, latency, or memory for the original results.

### Transformer Engine FP8 training release sequence

Keep the FP8 stack separate from both the shared project `.venv` and the private vLLM environment. The build uses the
private PyTorch 2.8.0/CUDA 12.4 wheel with Gloo/NCCL, pins Transformer Engine 2.8.0, and records an environment
manifest. Transformer Engine 2.8 requires cuDNN 9.3+, while the private Torch wheel reports its own cuDNN 8.9.7
runtime. The builder therefore installs a private cuDNN 9.3.0.75 package for Transformer Engine, together with
CUDA-12.4-matched cuBLAS 12.4.5.8 and NVRTC 12.4.127 libraries. It uses the matching UoP Spack NCCL 2.27.7
development root for compilation and pins Triton 3.4.0 for the PyTorch 2.8/Transformer Engine Python modules.
From a clean, current checkout on the login node:

```bash
source slurm/modules_for_hpc_ai
source .venv/bin/activate

NCCL_HEADER="$(find "${SPACK_INSTALL_ROOT:?Set SPACK_INSTALL_ROOT to the site package root}" \
  -type f -path '*/nccl-2.27.7-*/include/nccl.h' -print -quit)"
export NCCL_HOME="${NCCL_HEADER%/include/nccl.h}"
test -r "$NCCL_HOME/include/nccl.h"
test -r "$NCCL_HOME/lib/libnccl.so"

make submit-fp8-training-build NCCL_HOME="$NCCL_HOME"
# After the build job completes successfully:
make submit-fp8-training-acceptance PLATFORM=l40s
# After the acceptance job completes successfully:
make verify-fp8-training-acceptance PLATFORM=l40s

make submit-fp8-training-wave PLATFORM=l40s
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=training-fp8
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=training-fp8
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=training-fp8
```

The same private Torch/Transformer Engine build targets both L40S (`sm_89`) and H100 (`sm_90`). Run the H100
acceptance gate independently before the H100 controlled training-FP8 wave:

```bash
make submit-fp8-training-acceptance PLATFORM=h100
# After the H100 acceptance job completes successfully:
make verify-fp8-training-acceptance PLATFORM=h100
```

This writes separate diagnostic records below `build/fp8_acceptance/h100/` using H100-labelled configurations.
After it passes, submit the separately controlled canonical wave:

```bash
make submit-fp8-training-wave PLATFORM=h100
make status-platform-wave PLATFORM=h100 PLATFORM_WAVE=training-fp8
make collect-platform-wave PLATFORM=h100 PLATFORM_WAVE=training-fp8
make stage-platform-wave PLATFORM=h100 PLATFORM_WAVE=training-fp8
```

The H100 wave pairs GPT-2 `E086` with H100-only Transformer Engine BF16 control `E233`, and DistilBERT `E093`
with H100-only control `E234`. The platform resolver records those control-ID overrides in each resolved H100 config;
it never reuses the L40S `E231/E232` evidence.

The verifier permits an earlier successful acceptance record only when its commit is an ancestor of the current
checkout and `git diff` shows that the FP8 runtime contract (activation, smoke configurations, runner, workload, and
environment scripts) is unchanged. Adding controls or documentation therefore does not waste an H100 re-acceptance;
any runtime-contract change still requires a fresh gate.

`submit-fp8-training-build` installs the pinned private cuDNN/cuBLAS/NVRTC/Triton set inside the isolated
environment and passes the NCCL root as `HPC_AI_NCCL_HOME`. Its private library directories precede the system
toolkit at runtime, so the cuBLAS and cuBLASLt pair remain ABI-matched. The builder validates the NCCL header/shared
library, sets CMake/include/library paths, and records package/root/runtime versions in `BUILD_MANIFEST.json`. It
does not mutate the shared modules or the project `.venv`. It also generates `bin/activate_hpc_ai_fp8`; acceptance
and canonical FP8 wave jobs source that wrapper so their private CUDA-library search order matches the successful
build.

The acceptance JSON under `build/fp8_acceptance/` is diagnostic and must not be imported as canonical evidence. Each
canonical wave must report Transformer Engine modules for all four rows, `transformer_engine_bf16` for its controls,
and `transformer_engine_delayed_scaling_hybrid` plus `fp8_forward_enabled=true` for E086/E093. On L40S, E086/E093
pair with E231/E232; on H100 they pair with E233/E234. Existing E085/E092 PyTorch-AMP evidence remains valid for
its original condition but is not the controlled backend match for either FP8 comparison.

### Full-node L40S release sequence

Run the full-node waves in this order. Each status command must report all expected trials as provenance-valid before
collection and staging. The first four waves use the project `.venv`; the last two use the isolated vLLM environment.

```bash
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-pilot
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-pilot
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-pilot
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-pilot

make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-scaling
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-strategy
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=8gpu-communication

make submit-vllm-wave PLATFORM=l40s PLATFORM_WAVE=vllm-tp8-canary
make submit-vllm-wave PLATFORM=l40s PLATFORM_WAVE=vllm-tp8
```

For every performance wave, substitute its wave name in the same `status-platform-wave`, `collect-platform-wave`, and
`stage-platform-wave` sequence shown for `8gpu-pilot`. Submit `vllm-tp8` only after the E230 canary is collected.

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

The run-state file is ignored orchestration state and may be absent after moving to another checkout. If verified
evidence for part of the wave is already committed under `results/raw`, `retry-platform-wave` reconstructs completed
conditions from that evidence and submits only the missing conditions. Collection then combines the recovery bundle
with existing accepted evidence without rerunning or duplicating completed trials.

## 5. H100 Recovery Path

For H100 execution, verify the node and start or resume its independent waves:

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

Before using H100 power, utilisation, external-memory, or energy values in conclusions, run the independent H100
`telemetry-revalidation` wave and follow [the GPU telemetry revalidation protocol](gpu_telemetry_revalidation.md).

H100 vLLM uses the private environment, never the project `.venv`. After the diagnostic acceptance gate, collect a
canonical pilot before performance evidence:

```bash
make submit-vllm-pilot PLATFORM=h100
make status-platform-wave PLATFORM=h100 PLATFORM_WAVE=vllm-pilot
make collect-platform-wave PLATFORM=h100 PLATFORM_WAVE=vllm-pilot
make stage-platform-wave PLATFORM=h100 PLATFORM_WAVE=vllm-pilot
```

Then use `make submit-vllm-wave PLATFORM=h100 PLATFORM_WAVE=<wave>` in this order: `vllm-framework`,
`vllm-fp8`, `vllm-batching`, `vllm-concurrency`, `vllm-prompt`, `vllm-tp2`, and canary-gated `vllm-tp4`.
Run the separate Qwen `vllm-long-context` continuation through `make submit-long-context-acceptance` and
`make submit-long-context-wave`; its 8,192-token prefill gate is not interchangeable with E006 or the smaller-model
prompt sweep.

### H100 resource-homogeneous waves

H100 has four GPUs, but a reservation may expose fewer. The following waves deliberately separate collectable
one-/two-GPU work from full-node work; do not submit a full-node wave while only two GPUs are allocated.

| Wave | Conditions | GPU request per job | Trials |
| --- | --- | ---: | ---: |
| `4b-2gpu` | `E113`, `E114`, `E117`, `E118` | 2 | 3 each |
| `optional-1to2gpu` | `E016`, `E017`, `E044`, `E045` | 1 or 2 | 3 each |
| `4b-4gpu` | `E115`, `E116`, `E119`, `E120` | 4 | 3 each |
| `optional-4gpu` | `E046` | 4 | 3 |
| `vllm-tp4` | `E148`, `E196` | 4 | 3 each |

With a two-GPU H100 allocation, submit and collect the first two waves independently:

```bash
make submit-platform-wave PLATFORM=h100 PLATFORM_WAVE=4b-2gpu
make submit-platform-wave PLATFORM=h100 PLATFORM_WAVE=optional-1to2gpu
```

When a full four-GPU/64-CPU node allocation is available, submit `4b-4gpu`, `optional-4gpu`, and the private-vLLM
`vllm-tp4` wave. E172 remains separately deferred until that same full-node allocation is explicitly available.

## 6. Recovery Rules

- Failed trial: retain it, diagnose, and rerun the same repetition. The tracker counts only the latest record for that
  repetition and does not inflate the sample size.
- Invalid result: fix the execution/provenance issue and rerun; never hand-edit primary evidence.
- Conflicting import filename: keep both originals outside Git, identify why run IDs collided, and resolve the workflow
  before importing. The importer will not overwrite either side.
- H100 still down: continue only L40S planned waves. Record H100 rows as Ready/Deferred, not Failed or Complete.
- L40S memory mismatch: stop the exact condition and register a capacity-adjusted condition before changing batch or
  accumulation settings.
