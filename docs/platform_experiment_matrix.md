# Enhanced Platform Experiment Matrix

This matrix routes every current `E001-E200` experiment family to Mac M1, L40/L40S, and H100. It supplements, rather than replaces, [the authoritative experiment matrix](experiment_matrix.md).

**Legend:** <br/>
**`Direct`** means the logical row already represents that platform's hardware-comparison half. <br/>
**`Resolved`** means the platform materializer creates a generated companion config without editing the logical row. <br/>
**`Dev proxy`** means use a local smoke/dev config only; it is not dissertation hardware evidence. <br/>
**`No`** means do not run that matrix row on the platform. <br/>
**`Deferred`** means do not schedule it anywhere yet.<br/>

| Current IDs | Experiment family | Mac M1 | L40/L40S | H100 | Clear reason and constraint |
| --- | --- | --- | --- | --- | --- |
| `E001-E006` | CUDA, DDP/NCCL, FSDP, DeepSpeed, vLLM environment validation | No matrix run; use smoke/dev only | Resolved; wave 0, run first | Resolved; wave 0 when available | Mac cannot prove CUDA/NCCL/Slurm. Generated configs bind the logical row to the observed GPU, partition, CPU policy, and result root. |
| `E007` | TensorRT-LLM environment validation | No | Deferred until an L40S-compatible engine environment exists | H100 acceptance condition | The build/load/request adapter is implemented. It still needs a compatible saved engine, pinned local model, and target-GPU acceptance result. |
| `E008-E015`, `E018-E019` | GPT-2, DistilBERT, TinyLlama baseline trials | Dev proxy for `E008`, `E010`, `E012`, `E014`, `E018` | Resolved; wave 1, 3 trials | Resolved; wave 1, 3 trials | Mac validates adapters only. L40S/H100 establish separately tracked baseline evidence. |
| `E016-E017` | TinyLlama training baseline/repeat | No; local tiny-random inference only | Resolved optional wave after memory pilot | Resolved optional wave after memory pilot | Full TinyLlama training is optional and memory/queue sensitive; do not substitute the tiny local model. |
| `E020-E031` | Workload characterisation | Dev proxy only for DistilBERT shape checks | Resolved; wave 1 | Resolved; wave 1 | nvidia-smi, CUDA traces, memory, and data-loader timings need NVIDIA hardware. |
| `E032-E034`, `E038-E040`, `E044-E046` | Strong scaling | No | Resolved; wave 3 or optional after 1/2/4-GPU allocation check | Resolved when H100 returns | L40S results characterise PCIe-topology scaling; H100 results are separate H100/NVLink conditions. |
| `E035-E037`, `E041-E043` | Weak scaling | No | Resolved; wave 3 | Resolved; wave 3 | Requires concurrent GPUs; keep work-per-GPU definitions unchanged. |
| `E047-E062` | DDP, FSDP, and ZeRO strategy comparison | No | Resolved after the L40S NCCL/FSDP pilot | Resolved after the H100 pilot | Multi-process CUDA/NCCL/DeepSpeed execution is HPC-only. Interconnect differences are part of the result. |
| `E063-E082` | Memory optimisation: checkpointing, micro-batch, accumulation | Dev proxy only for GPT-2 micro-batch | Resolved; wave 4 | Resolved; wave 4 | Mac cannot report target GPU memory. Any capacity adjustment must remain a separately labelled condition and cannot be pooled as exact. |
| `E083-E085`, `E087-E088`, `E090-E092`, `E094-E095` | Supported FP32/FP16/BF16 precision study | No | Resolved; wave 4 | Resolved; wave 4 | Valid NVIDIA precision comparisons after target-platform validation. Mac runs fp32 dev paths only. |
| `E086`, `E089`, `E093`, `E096` | FP8 precision study | No | Deferred | Deferred | A selected, validated FP8 backend/model path is not yet implemented. |
| `E097-E112` | Data movement: workers, pinned memory, prefetch | No | Resolved; wave 4 | Resolved; wave 4 | Pinned-memory and transfer results are CUDA/node dependent; platform profiles preserve the different CPU allocation policies. |
| `E113-E120` | NCCL and FSDP communication analysis | No | Resolved wave 4b; analyse NCCL over PCIe | Resolved wave 4b; include NVLink evidence where available | Mac has no NCCL. L40S cannot produce H100/NVLink evidence, so the two condition sets are platform-specific. |
| `E121`, `E122`, `E125`, `E126` | H100 hardware-comparison half | No | No; use the L40S counterparts instead | Direct | These rows are explicitly H100-labelled and form matched pairs with `E123/E124/E127/E128`. |
| `E123`, `E124`, `E127`, `E128` | L40S hardware-comparison half | No | Direct | No; use the H100 counterparts instead | Existing L40S configs preserve the controlled H100-vs-L40S comparison design. Confirm actual GPU name/memory first. |
| `E129`, `E130`, `E153`, `E177`, `E178` | Supported inference-framework comparison | Dev proxy only for PyTorch adapter paths | Resolved wave 5 after vLLM pilot | Resolved wave 5 after vLLM pilot | PyTorch/vLLM engine outcomes require CUDA and serving dependencies; Mac runs only local adapter checks. |
| `E131`, `E154`, `E155`, `E179` | Unsupported/incompatible inference frameworks | No | Deferred | Deferred | E131/E179 still need performance-adapter integration; generic vLLM/TensorRT-LLM causal adapters are incompatible with the current DistilBERT classification path. |
| `E132-E137`, `E156-E161`, `E180-E185` | Inference batching | No | Resolved; wave 5 | Resolved; wave 5 | Batch-throughput/latency depends on the serving GPU and engine; preserve request shape and model revision. |
| `E138-E142`, `E162-E166`, `E186-E190` | Inference concurrency | No | Resolved; wave 5 | Resolved; wave 5 | Requires CUDA serving load. Do not treat local thread timing as production-serving evidence. |
| `E143-E146`, `E167-E170`, `E191-E194` | Prompt/input-length study | Dev proxy for GPT-2/TinyLlama generated-length mechanics | Resolved; wave 5 | Resolved; wave 5 | Mac verifies generation mechanics only; target latency and memory claims require the NVIDIA platform. |
| `E147-E148`, `E171-E172`, `E195-E196` | Multi-GPU inference | No | Resolved after L40S tensor-parallel pilot | Resolved after H100 tensor-parallel pilot | Requires 2/4 GPUs. Current rows do not claim unavailable request-path NCCL telemetry. |
| `E149-E150`, `E173-E174`, `E197-E198` | Supported inference precision: FP16/BF16 | No | Resolved; wave 5 | Resolved; wave 5 | CUDA/serving precision comparison; validate each engine on the target GPU. |
| `E151-E152`, `E175-E176`, `E199-E200` | FP8 and INT8 inference/quantization | No | Deferred | Deferred | FP8 backend or quantized/calibrated immutable artefact is not yet available. |

## Directly Runnable Config Sets

| Platform | Current configs you can run without changing target hardware labels | Important condition |
| --- | --- | --- |
| Mac M1 | `configs/local/*.json` (4 smoke configs) and `configs/dev/*.json` (15 dev configs) | Workflow validation only. Keep results out of the `E001-E200` evidence set. |
| L40/L40S | 181 planned `E###@l40s` conditions | Generated configs use `gpu_l40s`, 8 CPUs/GPU, PCIe, and `results/raw/l40s`. Nineteen rows are explicitly deferred. |
| H100 | 182 planned `E###@h100` conditions | Generated configs use `gpu_h100`, 16 CPUs/GPU, NVLink, and `results/raw/h100`. Eighteen rows are explicitly deferred. |

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
| 0 — prove the path | 4 smoke configs, then 15 dev configs | Resolved `E001-E006` | Resolved `E001-E007` when the node returns |
| 1 — core evidence | Dev proxies only | Wave 1 baselines/characterisation, direct hardware pairs, then wave 3 pilot | Wave 1 baselines/characterisation, direct hardware pairs, then wave 3 pilot |
| 2 — depth | Local shape checks | Waves 4, 4b, and 5 | Waves 4, 4b, and 5 |
| 3 — future work | Not applicable | Deferred rows only after prerequisites | Deferred rows only after prerequisites |
