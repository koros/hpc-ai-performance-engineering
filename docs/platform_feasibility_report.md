# Platform Feasibility and Experiment Routing Report

## Purpose

This report maps the current `E001-E200` experiment matrix to three execution targets:

- a local Apple Silicon Mac M1;
- the Lovelace L40/L40S GPU node; and
- the Lovelace H100 GPU node.

It is a scheduling and evidence-quality guide. It does not change the authoritative matrix or permit results from one platform to be labelled as another.

Operationally, the H100 node is currently unavailable. The active collection path is therefore L40S-first. The
same logical matrix and measurement protocol are retained so H100 conditions can be generated later without
rewriting source configs or mixing evidence.

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
| Total matrix configs | 200 | `E001-E200` |
| Planned L40S conditions | 181 | Generated as `E###@l40s`; run in gated waves while H100 is unavailable. |
| Deferred L40S conditions | 19 | Fourteen backend/adapter rows, L40S `E007`, and four H100-specific hardware rows. |
| Planned H100 conditions | 182 | Generated as `E###@h100` when the node returns. |
| Deferred H100 conditions | 18 | Fourteen backend/adapter rows and four L40S-specific hardware rows. |
| Mac M1 smoke configs | 4 | `configs/local/*`; minimal workflow checks. |
| Mac M1 dev configs | 15 | `configs/dev/*`; lightweight MPS/CPU development checks. |

The framework deliberately checks the requested GPU model at runtime and rejects an H100-labelled job on an
L40/L40S allocation (and vice versa). It also rejects CPU/MPS fallback for a GPU-labelled matrix run. The platform
materializer now creates the required companion config deterministically and records the logical-config SHA-256.

## Hardware and Scheduler Facts That Affect the Plan

The Lovelace documentation describes one four-GPU H100 node and one eight-GPU L40S node. The `gpu_h100` queue provides 16 CPU cores per allocated GPU, while `gpu_l40s` provides 8; do not copy an H100 scheduler allocation unchanged to L40S. [Lovelace system specification](https://www.plymouth.ac.uk/facilities/high-performance-computing) and [GPU queue guidance](https://docs.lovelace.plymouth.ac.uk/user-guide/resources) should be checked again immediately before submission.

The H100 SXM is the primary multi-GPU platform: it has 80 GB memory and NVLink. NVIDIA lists the L40S as a PCIe GPU with no NVLink. Consequently, an L40S distributed result is useful evidence about DDP/FSDP/ZeRO and NCCL over the installed PCIe topology, but it cannot support an H100 NVLink conclusion. [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and [NVIDIA L40S specifications](https://www.nvidia.com/en-in/data-center/l40s/) provide the hardware distinction.

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

L40S is suitable for single-GPU baselines, 1/2/4-GPU scaling, DDP/FSDP/ZeRO comparisons, memory optimisation, data-movement studies, power/energy analysis, and supported PyTorch/vLLM inference. Its results answer L40S-specific questions. They do not replace H100 evidence for H100-specific research questions.

### H100: deferred collection path, already materializable

When the node returns, generate the 182 planned H100 conditions from the same matrix and begin with wave 0. This is
the only target that supports the dissertation's current wording about one-to-four H100 scaling and H100 NVLink
behaviour. No L40S result is promoted or relabelled; matched conditions are compared only after both exist.

Use the H100 node for the core baseline trials, 1/2/4-GPU scaling, distributed-strategy comparison, communication analysis, and H100 half of the hardware comparison. Preserve the three independent trials required for core comparisons; a successful one-off run is an environment check, not final performance evidence.

## Deferred Rows: Do Not Schedule on Any Platform Yet

| IDs | Reason | Required before release |
| --- | --- | --- |
| `E007@l40s` | The saved-engine adapter is implemented, but an L40S-compatible TensorRT-LLM engine environment is not prepared. | A compatible pinned environment, local TinyLlama checkpoint, and successful build/load/request result. `E007@h100` remains an H100 acceptance condition. |
| `E131`, `E155`, `E179` | TensorRT-LLM performance integration is absent; `E155` is also not a valid generic DistilBERT classification adapter. | Reuse the validated E007 engine contract in causal-LM benchmarks; design a separate classification contract for E155. |
| `E086`, `E089`, `E093`, `E096`, `E151`, `E175`, `E199` | FP8 backend/model implementation is not selected or validated. | A documented, tested backend and model path on the target GPU. |
| `E152`, `E176`, `E200` | Quantization method, calibration, and immutable quantized artefact are absent. | Named method, calibration protocol, artefact hash, and matching serving runtime. |
| `E154` | vLLM is a causal-generation engine, not the current DistilBERT classification adapter. | A valid classification serving adapter or a redesign of the comparison. |

## Recommended Collection Order

| Order | Mac M1 | L40/L40S | H100 | Why |
| ---: | --- | --- | --- | --- |
| 0 | Smoke and dev profile | Inspect GPU name, memory, CUDA, and topology | Inspect GPU name, memory, CUDA, and topology | Avoid model/profile mismatch before spending queue time. |
| 1 | Baseline-shaped dev checks | Wave 0: `E001-E006` | H100 wave 0 when available | Establish CUDA, NCCL, FSDP, DeepSpeed, and serving viability. |
| 2 | GPT-2/DistilBERT adapter checks | Wave 1, then wave 2 direct hardware pairs | H100 wave 1, then wave 2 when available | Build repeatable reference measurements and matched platform anchors. |
| 3 | No further publishable work | Wave 3 scaling and strategy pilots | H100 wave 3 later | Check 1/2/4-GPU availability and strategy stability before large sweeps. |
| 4 | Optional local shape checks | Waves 4, 4b, and 5 | Matching H100 waves later | Collect depth only after the core path is stable. |

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
