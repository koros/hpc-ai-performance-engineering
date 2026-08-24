# Enhanced Platform Experiment Matrix

This matrix routes every current `E001-E200` experiment family to Mac M1, L40/L40S, and H100. It supplements, rather than replaces, [the authoritative experiment matrix](experiment_matrix.md).

Legend: **Direct** means use the existing config on the matching platform. **Retarget** means create a platform-specific companion config first. **Dev proxy** means use a local smoke/dev config only; it is not dissertation hardware evidence. **No** means do not run that matrix row on the platform. **Deferred** means do not schedule it anywhere yet.

| Current IDs | Experiment family | Mac M1 | L40/L40S | H100 | Clear reason and constraint |
| --- | --- | --- | --- | --- | --- |
| `E001-E006` | CUDA, DDP/NCCL, FSDP, DeepSpeed, vLLM environment validation | No matrix run; use smoke/dev only | Retarget, then run first | Direct, run first | The current configs explicitly require H100 CUDA. L40S needs platform/model/queue retargeting; Mac cannot prove CUDA/NCCL/Slurm. |
| `E007` | TensorRT-LLM environment validation | No | Deferred | Deferred | Built TensorRT-LLM engine and adapter contract are missing. |
| `E008-E015`, `E018-E019` | GPT-2, DistilBERT, TinyLlama inference baseline trials | Dev proxy for `E008`, `E010`, `E012`, `E014`, `E018` | Retarget; run 3 independent trials | Direct; run 3 independent trials | Mac validates adapters only. L40S/H100 establish platform-specific baseline evidence. |
| `E016-E017` | TinyLlama training baseline/repeat | No; local tiny-random inference only | Retarget after memory pilot | Direct after memory pilot | Full TinyLlama training is optional and memory/queue sensitive; do not substitute the tiny local model. |
| `E020-E031` | Workload characterisation | Dev proxy only for DistilBERT shape checks | Retarget | Direct | nvidia-smi, CUDA traces, memory, and data-loader timings need NVIDIA hardware. |
| `E032-E034`, `E038-E040`, `E044-E046` | Strong scaling | No | Retarget if 1/2/4 L40S GPUs can be allocated | Direct | Requires concurrent GPUs; L40S results are PCIe-topology scaling, H100 results answer the current H100 RQ. |
| `E035-E037`, `E041-E043` | Weak scaling | No | Retarget if 1/2/4 L40S GPUs can be allocated | Direct | Same multi-GPU requirement; keep work-per-GPU definitions unchanged. |
| `E047-E062` | DDP, FSDP, and ZeRO strategy comparison | No | Retarget after L40S NCCL/FSDP pilot | Direct after H100 pilot | Multi-process CUDA/NCCL/DeepSpeed execution is HPC-only. Interconnect differences are part of the result. |
| `E063-E082` | Memory optimisation: checkpointing, micro-batch, accumulation | Dev proxy only for GPT-2 micro-batch | Retarget; begin at smaller batch if required | Direct | Mac cannot report target GPU memory. L40S has less/uncertain reported memory, so retuning changes the condition and must be documented. |
| `E083-E085`, `E087-E088`, `E090-E092`, `E094-E095` | Supported FP32/FP16/BF16 precision study | No | Retarget | Direct | Valid NVIDIA precision comparisons after target-platform validation. Mac runs fp32 dev paths only. |
| `E086`, `E089`, `E093`, `E096` | FP8 precision study | No | Deferred | Deferred | A selected, validated FP8 backend/model path is not yet implemented. |
| `E097-E112` | Data movement: workers, pinned memory, prefetch | No | Retarget | Direct | Pinned-memory and transfer results are CUDA/node dependent; use the L40S CPU allocation policy. |
| `E113-E120` | NCCL and FSDP communication analysis | No | Retarget; analyse NCCL over PCIe | Direct; include NVLink evidence where available | Mac has no NCCL. L40S has no NVLink, so do not request or infer NVLink utilisation. |
| `E121`, `E122`, `E125`, `E126` | H100 hardware-comparison half | No | No; use the L40S counterparts instead | Direct | These rows are explicitly H100-labelled and form matched pairs with `E123/E124/E127/E128`. |
| `E123`, `E124`, `E127`, `E128` | L40S hardware-comparison half | No | Direct | No; use the H100 counterparts instead | Existing L40S configs preserve the controlled H100-vs-L40S comparison design. Confirm actual GPU name/memory first. |
| `E129`, `E130`, `E153`, `E177`, `E178` | Supported inference-framework comparison | Dev proxy only for PyTorch adapter paths | Retarget after vLLM pilot | Direct after vLLM pilot | PyTorch/vLLM engine outcomes require CUDA and serving dependencies; Mac runs only local adapter checks. |
| `E131`, `E154`, `E155`, `E179` | Unsupported/incompatible inference frameworks | No | Deferred | Deferred | TensorRT engine contracts are absent; generic vLLM is incompatible with current DistilBERT classification path. |
| `E132-E137`, `E156-E161`, `E180-E185` | Inference batching | No | Retarget | Direct | Batch-throughput/latency depends on the serving GPU and engine; preserve request shape and model revision. |
| `E138-E142`, `E162-E166`, `E186-E190` | Inference concurrency | No | Retarget | Direct | Requires CUDA serving load. Do not treat local thread timing as production-serving evidence. |
| `E143-E146`, `E167-E170`, `E191-E194` | Prompt/input-length study | Dev proxy for GPT-2/TinyLlama generated-length mechanics | Retarget | Direct | Mac verifies generation mechanics only; target latency and memory claims require the NVIDIA platform. |
| `E147-E148`, `E171-E172`, `E195-E196` | Multi-GPU inference | No | Retarget after L40S tensor-parallel pilot | Direct after H100 tensor-parallel pilot | Requires 2/4 GPUs. Current rows do not claim unavailable request-path NCCL telemetry. |
| `E149-E150`, `E173-E174`, `E197-E198` | Supported inference precision: FP16/BF16 | No | Retarget | Direct | CUDA/serving precision comparison; validate each engine on the target GPU. |
| `E151-E152`, `E175-E176`, `E199-E200` | FP8 and INT8 inference/quantization | No | Deferred | Deferred | FP8 backend or quantized/calibrated immutable artefact is not yet available. |

## Directly Runnable Config Sets

| Platform | Current configs you can run without changing target hardware labels | Important condition |
| --- | --- | --- |
| Mac M1 | `configs/local/*.json` (4 smoke configs) and `configs/dev/*.json` (15 dev configs) | Workflow validation only. Keep results out of the `E001-E200` evidence set. |
| L40/L40S | `E123`, `E124`, `E127`, `E128` | Confirm the allocated device is truly `L40S`; use the `gpu_l40s` queue and L40S profile. |
| H100 | All non-deferred H100-labelled rows: 181 configs | Run `E001-E006` first and collect the required independent trials for core comparisons. |

## Retargeting Checklist for L40/L40S Companion Experiments

1. Copy the H100 source config; retain the original unchanged.
2. Give the copy a platform-specific identity and record the source experiment ID.
3. Set the observed L40S platform/model and PCIe interconnect.
4. Use the L40S Slurm queue/profile; its CPU allocation differs from H100.
5. Run the relevant L40S environment validation before the workload family.
6. Retain workload, model revision, data, seed, and global-work protocol whenever memory allows.
7. If memory requires a batch/accumulation change, label it as a new L40S condition and do not pool it with equal-work H100 results.
8. Record `nvidia-smi` name, memory, driver, CUDA, and topology in the raw result.

## Priority Tiers

| Tier | Mac M1 | L40/L40S | H100 |
| --- | --- | --- | --- |
| 0 — prove the path | 4 smoke configs, then 15 dev configs | Retargeted `E001-E006` | `E001-E006` |
| 1 — core evidence | Dev proxies only | Retargeted baselines, then 1/2/4-GPU pilot | H100 baselines, scaling, and strategy pilots |
| 2 — depth | Local shape checks | Memory, data movement, communication, supported inference | Full supported matrix excluding deferred rows |
| 3 — future work | Not applicable | Deferred rows only after prerequisites | Deferred rows only after prerequisites |
