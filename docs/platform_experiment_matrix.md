# Enhanced Platform Experiment Matrix

This matrix routes every current `E001-E200` experiment family to Mac M1, L40/L40S, and H100. It supplements, rather than replaces, [the authoritative experiment matrix](experiment_matrix.md).

**Legend:** <br/>
**`Direct`** means the logical row already represents that platform's hardware-comparison half. <br/>
**`Resolved`** means the platform materializer creates a generated companion config without editing the logical row. <br/>
**`Dev proxy`** means use a local smoke/dev config only; it is not dissertation hardware evidence. <br/>
**`No`** means do not run that matrix row on the platform. <br/>
**`Deferred`** means do not schedule that condition on the named platform yet.<br/>

Current site policy: the PyTorch CUDA runtime is usable on both GPU partitions, so ordinary CUDA/NCCL/FSDP/DeepSpeed
experiments remain scheduled. The private PyTorch 2.8.0/CUDA 12.4 and vLLM 0.10.2 stack has passed E006 on L40S,
so supported L40S vLLM conditions use dedicated waves. H100 vLLM waits for the same target-platform gate, while
TensorRT-LLM remains deferred until its package, engine artefact, and load/request contract exist.

| Current IDs | Experiment family | Mac M1 | L40/L40S | H100 | Clear reason and constraint |
| --- | --- | --- | --- | --- | --- |
| `E001`, `E002`, `E004-E005` | CUDA, DDP/NCCL, FSDP, DeepSpeed environment validation | No matrix run; use smoke/dev only | Resolved; wave 0, run first | Resolved; wave 0, run first | Mac cannot prove CUDA/NCCL/Slurm. Generated configs bind the logical row to the observed GPU, partition, CPU policy, and result root. |
| `E003` | Four-GPU NCCL validation | No matrix run | Deferred | Resolved when four H100 GPUs are available | The active L40S node exposes two GPUs; four-GPU evidence must not be substituted with a two-GPU run. |
| `E006` | vLLM environment validation | No | Resolved; `vllm-pilot`, 1 trial | Deferred | The private Gloo/NCCL-enabled PyTorch and vLLM stack passed import, engine-start, and served-request validation on L40S. H100 still needs the same acceptance gate. |
| `E007` | TensorRT-LLM environment validation | No | Deferred | Deferred | CUDA 12.4.1 is validated, but the compatible TensorRT-LLM package, built engine artefact, pinned model, and load/request contract are still missing. |
| `E008-E015`, `E018-E019` | GPT-2, DistilBERT, TinyLlama baseline trials | Dev proxy for `E008`, `E010`, `E012`, `E014`, `E018` | Resolved; wave 1, 3 trials | Resolved; wave 1, 3 trials | Mac validates adapters only. L40S/H100 establish separately tracked baseline evidence. |
| `E016-E017` | TinyLlama training baseline/repeat | No; local tiny-random inference only | Resolved optional wave after memory pilot | Resolved optional wave after memory pilot | Full TinyLlama training is optional and memory/queue sensitive; do not substitute the tiny local model. |
| `E020-E031` | Workload characterisation | Dev proxy only for DistilBERT shape checks | Resolved; wave 1 | Resolved; wave 1 | nvidia-smi, CUDA traces, memory, and data-loader timings need NVIDIA hardware. |
| `E032-E033` | GPT-2 Strong scaling exact shapes | No | Deferred | Resolved when H100 returns | The exact L40S shapes OOMed on the observed 44.52 GiB devices: E032 after allocating 43.32 GiB, and E033 before an additional 6.14 GiB allocation. Smaller L40S variants must be separate capacity-adjusted conditions. |
| `E034`, `E038-E040`, `E044-E046` | Strong scaling | No | Run supported DistilBERT rows and E045; defer E044 after measured OOM and four-GPU rows | Resolved when H100 returns | E044's exact one-GPU TinyLlama batch-16 shape OOMed in all three L40S trials, while E045 completed on two GPUs. Original E044-E046 remain unchanged for H100. |
| `E035-E037`, `E041-E043` | Weak scaling | No | Resolved; wave 3 | Resolved; wave 3 | Requires concurrent GPUs; keep work-per-GPU definitions unchanged. |
| `E047-E062` | DDP, FSDP, and ZeRO strategy comparison | No | Resolved after the L40S NCCL/FSDP pilot | Resolved after the H100 pilot | Multi-process CUDA/NCCL/DeepSpeed execution is HPC-only. Interconnect differences are part of the result. |
| `E063-E082` | Memory optimisation: checkpointing, micro-batch, accumulation | Dev proxy only for GPT-2 micro-batch | Resolved exact wave 4 except E067/E068; those are deferred | Resolved; wave 4 | Mac cannot report target GPU memory. E067/E068 exact GPT-2 micro-batch 32/64 rows OOM on L40S and remain H100-exact conditions. |
| `E201-E207` | L40S capacity-adjusted GPT-2 micro-batch boundary probes | No | Resolved wave 4c for E201/E202/E205; E203/E204/E206/E207 deferred after OOM | No | These supplemental rows test L40S batch sizes 20/24/25/26/27 after E067/E068 exact OOM and E203/E204 upper-bound OOM at 28/30. E205 at micro-batch 25 is the largest passing integer probe observed so far; E206/E207 establish the immediate failing boundary at 26/27. They answer the L40S capacity boundary question and must not be pooled with exact H100 E067/E068 evidence. |
| `E208-E209` | L40S checkpointed TinyLlama strong scaling | No | Resolved; `optional-capacity` | No | This matched one-/two-GPU pair enables activation checkpointing on both rows while preserving BF16, sequence length 1024, global batch 16, and the original per-GPU batch shapes. It is capacity-adjusted evidence and is not pooled with exact E044/E045/H100 results. |
| `E210-E213` | GPU telemetry revalidation anchors | No | Resolved; `telemetry-revalidation` | Resolved; `telemetry-revalidation` | Separate one-/two-GPU training, inference, and communication anchors verify UUID-bound nvidia-smi attribution on each platform. They are `platform_specific` diagnostic evidence and do not replace or pool with their reference rows. |
| `E083-E085`, `E087-E088`, `E090-E092`, `E094-E095` | Supported FP32/FP16/BF16 precision study | No | Resolved; wave 4 | Resolved; wave 4 | Valid NVIDIA precision comparisons after target-platform validation. Mac runs fp32 dev paths only. |
| `E086`, `E089`, `E093`, `E096` | FP8 precision study | No | Deferred | Deferred | A selected, validated FP8 backend/model path is not yet implemented. |
| `E097-E112` | Data movement: workers, pinned memory, prefetch | No | Resolved; wave 4 | Resolved; wave 4 | Pinned-memory and transfer results are CUDA/node dependent; platform profiles preserve the different CPU allocation policies. |
| `E113-E120` | NCCL and FSDP communication analysis | No | Resolved wave 4b; analyse NCCL over PCIe | Resolved wave 4b; include NVLink evidence where available | Mac has no NCCL. L40S cannot produce H100/NVLink evidence, so the two condition sets are platform-specific. |
| `E121`, `E122`, `E125`, `E126` | H100 hardware-comparison half | No | No; use the L40S counterparts instead | Direct | These rows are explicitly H100-labelled and form matched pairs with `E123/E124/E127/E128`. |
| `E123`, `E124`, `E127`, `E128` | L40S hardware-comparison half | No | Direct | No; use the H100 counterparts instead | Existing L40S configs preserve the controlled H100-vs-L40S comparison design. Confirm actual GPU name/memory first. |
| `E129`, `E149-E150`, `E153`, `E156-E174`, `E177`, `E197-E198` | Supported PyTorch inference | Dev proxy only for PyTorch adapter paths | Resolved; wave 5 | Resolved; wave 5 | These rows use the validated PyTorch CUDA runtime and do not depend on vLLM or a CUDA compiler. |
| `E130`, `E132-E148`, `E178`, `E180-E196` | vLLM serving, batching, concurrency, prompt, and tensor-parallel studies | No | Resolved in dedicated `vllm-*` waves except four-GPU `E148/E196` | Deferred | The accepted private L40S environment supports 34 exact one-/two-GPU performance conditions. E148/E196 exceed the active two-GPU L40S node; H100 rows wait for H100 E006. Do not substitute PyTorch results for vLLM evidence. |
| `E131`, `E154`, `E155`, `E179` | Unsupported/incompatible inference frameworks | No | Deferred | Deferred | E131/E179 still need performance-adapter integration; generic vLLM/TensorRT-LLM causal adapters are incompatible with the current DistilBERT classification path. |
| `E156-E161`, `E162-E174` | DistilBERT PyTorch batching, concurrency, prompt, and multi-GPU studies | No | Resolved; wave 5 | Resolved; wave 5 | These are classification-compatible PyTorch runs; preserve request shape and GPU count. |
| `E149-E150`, `E173-E174`, `E197-E198` | Supported inference precision: FP16/BF16 | No | Resolved; wave 5 | Resolved; wave 5 | CUDA/serving precision comparison; validate each engine on the target GPU. |
| `E151-E152`, `E175-E176`, `E199-E200` | FP8 and INT8 inference/quantization | No | Deferred | Deferred | FP8 backend or quantized/calibrated immutable artefact is not yet available. |

## Directly Runnable Config Sets

| Platform | Current configs you can run without changing target hardware labels | Important condition |
| --- | --- | --- |
| Mac M1 | `configs/local/*.json` (4 smoke configs) and `configs/dev/*.json` (15 dev configs) | Workflow validation only. Keep results out of the `E001-E200` evidence set. |
| L40/L40S | 159 planned source `E###@l40s` conditions, 5 planned capacity-adjusted conditions, and 4 telemetry anchors | Generated configs use `gpu_l40s`, 8 CPUs/GPU, PCIe, and accepted evidence root `results/raw/l40s`. Forty-one source rows remain deferred: TensorRT-LLM/backend gaps, platform-specific comparison rows, four-GPU rows, and exact-shape E032/E033/E044/E067/E068 OOMs. E203/E204/E206/E207 are supplemental upper-bound OOM deferrals. |
| H100 | 144 planned source `E###@h100` conditions plus 4 telemetry anchors | Generated configs use `gpu_h100`, 16 CPUs/GPU, NVLink, and accepted evidence root `results/raw/h100`. Fifty-six source rows remain deferred, including 37 vLLM conditions pending H100 E006, TensorRT-LLM, backend/adapter gaps, and L40S-specific comparison rows. |

## Generated Companion-Config Contract

1. Keep `configs/` as the immutable logical matrix; do not hand-edit an H100 row into an L40S row.
2. Resolve it with `materialize-platform-configs` or `render-platform-slurm` and a tracked platform profile.
3. Preserve logical `experiment_id: E###`; use `condition_id: E###@l40s` or `E###@h100` for evidence identity.
4. Record the source config path and SHA-256, execution wave, platform, result root, and comparability class.
5. Retain workload, model revision, data, seed, and global-work protocol whenever memory allows.
6. If capacity requires a batch/accumulation change, register it as `capacity_adjusted`; never silently pool it with `exact` conditions.
7. Record the observed GPU name, memory, driver, CUDA, and topology in every accepted raw result.

## Priority Tiers

| Tier | Mac M1 | L40/L40S | H100 |
| --- | --- | --- | --- |
| 0 — prove the path | 4 smoke configs, then 15 dev configs | Resolved `E001`, `E002`, `E004-E005`; defer `E003` | Resolved `E001-E005` |
| 1 — core evidence | Dev proxies only | Wave 1 baselines/characterisation, direct hardware pairs, then wave 3 pilot | Wave 1 baselines/characterisation, direct hardware pairs, then wave 3 pilot |
| 2 — depth | Local shape checks | Waves 4, 4c, 4b, supported-PyTorch wave 5, then `vllm-pilot`, `vllm-framework`, `vllm-batching`, `vllm-concurrency`, `vllm-prompt`, and `vllm-tp2` | Waves 4, 4b, and supported-PyTorch wave 5 |
| 3 — future work | Not applicable | Four-GPU vLLM, TensorRT-LLM, FP8/INT8, and other deferred rows only after prerequisites | vLLM after H100 E006; other deferred rows only after prerequisites |
