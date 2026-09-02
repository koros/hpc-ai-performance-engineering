# Platform Feasibility and Experiment Routing Report

## Purpose

This report maps the current `E001-E200` experiment matrix to three execution targets:

- a local Apple Silicon Mac M1;
- the Lovelace L40/L40S GPU node; and
- the Lovelace H100 GPU node.

It is a scheduling and evidence-quality guide. It does not change the authoritative matrix or permit results from one platform to be labelled as another.

Operationally, both GPU targets are materializable and remain separate evidence streams. L40S collection is further
advanced and now includes an accepted private vLLM environment. H100 collection can proceed independently, but its
vLLM conditions remain deferred until the same E006 acceptance gate passes on H100.

The companion [platform experiment matrix](platform_experiment_matrix.md) gives the phase-by-phase routing decision for every matrix ID.

## Evidence Classes

| Class | Meaning | May support dissertation hardware-performance claims? |
| --- | --- | --- |
| Local development | A smoke or `dev_` configuration runs on Mac M1 using MPS or CPU. | No. Use it to verify code, models, result writing, aggregation, and dashboard inputs. |
| Direct platform run | The current config already requests the allocated GPU model and can be submitted unchanged after the relevant environment check. | Yes, provided the measurement protocol and repetitions are followed. |
| Resolved platform condition | A generated config binds a logical `E###` row to `E###@l40s` or `E###@h100`. | Yes, but only for the recorded platform and declared comparability class. Generated configs never overwrite the logical matrix. |
| Deferred | The current row requires a missing backend, engine, quantized artefact, or incompatible model/backend adapter. | No, until its stated prerequisite is met. |

## Current Matrix Inventory

| Scope | Count | Interpretation |
| --- | ---: | --- |
| Source matrix configs | 200 | `E001-E200` |
| Supplemental L40S capacity conditions | 9 | `E201-E207` are GPT-2 micro-batch boundary probes; `E208-E209` are a matched checkpointed TinyLlama strong-scaling pair derived from the E044 L40S OOM. |
| Supplemental telemetry conditions | 5 | `E210-E214` are allocation/measurement-bound revalidation anchors; they remain platform-specific. |
| Supplemental L40S full-node conditions | 16 | `E215-E230` capture the eight-rank gate, scaling, DDP/FSDP, PCIe communication, and TinyLlama TP=8 serving. They do not alter source or H100 rows. |
| Planned L40S source conditions | 176 | Generated as `E###@l40s`; includes all 37 platform-specific/exact vLLM conditions in dedicated waves and 17 newly released four-GPU rows. |
| Planned L40S supplemental conditions | 26 | Five capacity-adjusted rows, five telemetry anchors, and sixteen full-node rows. Do not pool capacity-adjusted or L40S-only evidence with exact H100 evidence. |
| Deferred L40S supplemental conditions | 4 | `E203-E204` and `E206-E207`; confirmed upper-bound OOM probes at micro-batch 26/27/28/30. |
| Deferred L40S source conditions | 24 | TensorRT-LLM/backend/adapter gaps, H100-specific hardware rows, and exact-shape E032/E033/E044/E067/E068 L40S OOMs. |
| Planned H100 conditions | 144 | Generated as `E###@h100`; run the same supported scope when allocation is available. |
| Deferred H100 conditions | 56 | Backend/adapter rows, 37 vLLM rows pending H100 E006, TensorRT-LLM `E007`, and four L40S-specific hardware rows. |
| Mac M1 smoke configs | 4 | `configs/local/*`; minimal workflow checks. |
| Mac M1 dev configs | 15 | `configs/dev/*`; lightweight MPS/CPU development checks. |

The framework deliberately checks the requested GPU model at runtime and rejects an H100-labelled job on an
L40/L40S allocation (and vice versa). It also rejects CPU/MPS fallback for a GPU-labelled matrix run. The platform
materializer now creates the required companion config deterministically and records the logical-config SHA-256.

## Hardware and Scheduler Facts That Affect the Plan

The Lovelace documentation describes one four-GPU H100 node and one eight-GPU L40S node. The `gpu_h100` queue provides 16 CPU cores per allocated GPU, while `gpu_l40s` provides 8; do not copy an H100 scheduler allocation unchanged to L40S. [Lovelace system specification](https://www.plymouth.ac.uk/facilities/high-performance-computing) and [GPU queue guidance](https://docs.lovelace.plymouth.ac.uk/user-guide/resources) should be checked again immediately before submission.

The H100 SXM has 80 GB memory and NVLink, while the L40S node provides eight PCIe GPUs without NVLink. Consequently, both platforms support four-GPU experiments, while only the L40S extension uses an eight-GPU full-node allocation. Their communication evidence answers different topology questions: L40S measures the installed PCIe path and H100 measures NVLink where available. E215 records the actual eight-GPU L40S topology before any full-node interpretation. [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [NVIDIA L40S specifications](https://www.nvidia.com/en-in/data-center/l40s/) provide the hardware distinction.

There is a memory-label discrepancy to resolve before the first L40S run: the repository and L40S config names expect `NVIDIA L40S 48GB`, while the University facility page currently says 40 GB per L40S card. Run `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` on the allocation and use the observed name and capacity in the companion profile/config. Do not assume an L40 and an L40S are interchangeable; the runtime currently matches the `l40s` model token.

## Platform Decisions

### Mac M1: development and workflow validation only

Run the four smoke configs in `configs/local/` and the 15 `configs/dev/` rows. They use `hardware.gpus: 0`, `fp32`, and `device: auto`, which selects MPS when available and CPU otherwise. They exercise realistic model-loading, training/inference adapters, result records, aggregation, and dashboard plumbing without pretending to measure NVIDIA hardware.

Recommended Mac M1 sequence:

1. `configs/local/synthetic_smoke.json` first: no ML dependency, validates result writing.
2. `configs/local/gpt2_small_smoke.json`, `configs/local/distilbert_smoke.json`, and `configs/local/tinyllama_inference_smoke.json` after installing workload extras.
3. `make run-dev-baselines` for local model-path checks.
4. `make run-dev-refresh` only after the baseline-shaped dev runs work.

The Mac equivalents are intentionally limited: `E008`, `E010`, `E012`, `E014`, and `E018` have baseline-shaped dev proxies; GPT-2 micro-batch and generated-token-length rows have small dev proxies; DistilBERT sequence-length rows have small dev proxies. No matrix distributed, NCCL, power, GPU-hour, vLLM, TensorRT-LLM, FP8, quantization, or hardware-comparison result may be claimed from Mac M1.

### L40/L40S: active NVIDIA evidence path

Run wave 0 first, then the core baselines/characterisation and the direct L40S hardware-comparison rows. Generated
companions are written under `build/platform_configs/l40s`; the logical `configs/` rows remain unchanged. Each
companion:

- records `condition_id: E###@l40s`, the source path/hash, wave, required trials, and comparability;
- names the target as `lovelace_l40s`, `NVIDIA L40S 48GB`, and PCIe, with runtime verification of the actual device;
- use `PCIe` as the interconnect unless topology inspection proves otherwise;
- retain model revision, dataset, seed, workload shape, and measurement protocol;
- lower micro-batch size or add accumulation only if the 48/observed-GB memory limit requires it, recording the change as a new experimental condition;
- use the `gpu_l40s` queue/profile and repeat the L40S environment-validation sequence before matrix collection; and
- report communication as NCCL/PCIe behaviour, not NVLink behaviour.

L40S is suitable for supported single-, two-, four-, and—after E215—eight-GPU baselines, scaling, DDP/FSDP/ZeRO comparisons, communication analysis, memory optimisation, data-movement studies, power/energy analysis, PyTorch inference, and the accepted vLLM scope. Slurm job 1773321 verified four concurrent visible L40S GPUs on the eight-GPU node; E215 is the required canonical eight-rank confirmation. E216-E228 use the full node for scaling, strategy, and PCIe communication, while E230 gates E229 TinyLlama TP=8. The exact E032/E033 GPT-2 strong-scaling shapes and E067/E068 GPT-2 micro-batch 32/64 shapes remain deferred after observed OOMs; E201/E202/E205 provide separately labelled L40S capacity-adjusted probes, while E203/E204/E206/E207 record the upper-bound OOMs at micro-batch 26/27/28/30. E205 establishes micro-batch 25 as the largest passing integer boundary observed so far. Exact E044 also OOMed in all three L40S trials after allocating 43.30 GiB and failing a further 1000 MiB request. The matched checkpointed E208/E209 pair preserves TinyLlama, BF16, sequence length 1024, and global batch 16 for a separate capacity-adjusted scaling analysis, while original E044/E045 remain intact for H100. These results answer L40S-specific questions and do not replace H100 evidence for H100-specific research questions.

### H100: deferred collection path, already materializable

Generate the 144 planned H100 conditions from the same matrix and begin with wave 0. This is
the only target that supports the dissertation's current wording about one-to-four H100 scaling and H100 NVLink
behaviour. No L40S result is promoted or relabelled; matched conditions are compared only after both exist.

Use the H100 node for the core baseline trials, 1/2/4-GPU scaling, distributed-strategy comparison, communication analysis, and H100 half of the hardware comparison. Preserve the three independent trials required for core comparisons; a successful one-off run is an environment check, not final performance evidence.

## Remaining Deferred Rows

| IDs | Reason | Required before release |
| --- | --- | --- |
| H100 `E006`, `E130`, `E132-E148`, `E178`, `E180-E196` | The private stack has passed on L40S, not H100. | Run E006 with the private stack on H100 and require import, engine-start, and served-request success before releasing H100 performance rows. |
| `E007` | CUDA 12.4.1 is available, but TensorRT-LLM still lacks a compatible package, built engine, pinned model, and saved-engine contract. | A compatible pinned environment, immutable engine artefact, and successful build/load/request result on each target platform. |
| `E131`, `E155`, `E179` | TensorRT-LLM performance integration is absent; `E155` is also not a valid generic DistilBERT classification adapter. | Reuse the validated E007 engine contract in causal-LM benchmarks; design a separate classification contract for E155. |
| `E086`, `E089`, `E093`, `E096`, `E151`, `E175`, `E199` | FP8 backend/model implementation is not selected or validated. | A documented, tested backend and model path on the target GPU. |
| `E152`, `E176`, `E200` | Quantization method, calibration, and immutable quantized artefact are absent. | Named method, calibration protocol, artefact hash, and matching serving runtime. |
| `E154` | vLLM is a causal-generation engine, not the current DistilBERT classification adapter. | A valid classification serving adapter or a redesign of the comparison. |

## Recommended Collection Order

| Order | Mac M1 | L40/L40S | H100 | Why |
| ---: | --- | --- | --- | --- |
| 0 | Smoke and dev profile | Inspect GPU name, memory, CUDA, and topology | Inspect GPU name, memory, CUDA, and topology | Avoid model/profile mismatch before spending queue time. |
| 1 | Baseline-shaped dev checks | Wave 0: `E001`, `E002`, `E004-E005`; then `4gpu-pilot` E003 | Wave 0: `E001-E005` | Keep E003 separate so the completed core-wave history remains immutable and four-rank NCCL is explicitly gated. |
| 2 | GPT-2/DistilBERT adapter checks | Wave 1, then wave 2 direct hardware pairs | H100 wave 1, then wave 2 when available | Build repeatable reference measurements and matched platform anchors. |
| 3 | No further publishable work | `4gpu-core`, then `4gpu-communication` after E003; later E215 and the eight-GPU scaling/strategy/communication extension | H100 wave 3 later | Four-GPU L40S capacity is verified. E215 establishes the full-node topology before eight-GPU data is interpreted; exact E032/E033 OOM conditions remain deferred independently. |
| 4 | Optional local shape checks | Waves 4, 4c, 4b, and 5 | Matching H100 waves | Collect exact depth first, then labelled L40S capacity probes, communication, and PyTorch inference. |
| 5 | No publishable vLLM evidence | `vllm-pilot`, then framework, batching, concurrency, prompt, TP2, canary-gated TP4, and E230-gated TinyLlama TP8 | Run H100 E006 first; release H100 vLLM waves only after it passes | Keep private-runtime evidence platform-specific and preserve the accepted environment contract in each immutable bundle. |

## Interpretation Rules

- Keep Mac, L40S, and H100 records separate in analysis, figures, and dashboard filters.
- Compare H100 and L40S only for explicitly paired, identical workload settings; current pairs are `E121/E123`, `E122/E124`, `E125/E127`, and `E126/E128`.
- A resolved L40S config is a new condition. Do not overwrite an H100 raw result, configuration hash, or experiment label.
- If an L40S memory adjustment changes effective global batch or sequence length, it answers a capacity-aware L40S question, not the original equal-work H100 comparison.
- Report L40S communication limitations explicitly: it has no NVLink, so NVLink counters and conclusions are unavailable.
- Keep every platform condition marked Deferred out of batch generation until its prerequisite is recorded and the execution policy is updated explicitly.

## Before Each Platform Submission

```bash
python -m hpc_ai_perf.cli validate --skip-results --research-ready
python -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
make render-l40s-wave PLATFORM_WAVE=0
make track-l40s-progress
```

On a GPU allocation, record the exact device before collecting evidence:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

For Mac M1, use the local/dev targets rather than a GPU-labelled `E` config:

```bash
make run-dev-refresh
```
