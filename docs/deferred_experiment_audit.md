# Deferred Experiment Audit

Audit date: 2026-09-12.

The private PyTorch 2.8.0/CUDA 12.4 build provides both Gloo and NCCL, and the pinned vLLM 0.10.2 environment has now
passed E006 import, engine-start, and served-request acceptance on both L40S and H100. This resolves the former
platform-wide vLLM environment blockers. E151 and E199 now also have an explicit online vLLM FP8 W8A8 path. It does
not remove independent hardware, memory-capacity, TensorRT-LLM, INT8, or classification-adapter
blockers.

## Released From Deferred State

| Platform | IDs | New execution state | Reason |
| --- | --- | --- | --- |
| L40S | `E006` | Planned; `vllm-pilot`, 1 canonical trial | The diagnostic acceptance gate passed; the immutable pilot is still required as canonical matrix evidence. |
| L40S | `E130`, `E178` | Planned; `vllm-framework`, 3 trials | One-GPU GPT-2 and TinyLlama vLLM framework baselines now use the accepted private environment. |
| L40S | `E132-E137`, `E180-E185` | Planned; `vllm-batching`, 3 trials | One-GPU batching studies are supported by the accepted adapter/runtime. |
| L40S | `E138-E142`, `E186-E190` | Planned; `vllm-concurrency`, 3 trials | One-GPU concurrency studies are supported by the accepted adapter/runtime. |
| L40S | `E143-E146`, `E191-E194` | Planned; `vllm-prompt`, 3 trials | One-GPU prompt-length studies are supported by the accepted adapter/runtime. |
| L40S | `E147`, `E195` | Planned; `vllm-tp2`, 3 trials | The L40S node exposes the two GPUs required by these tensor-parallel rows. |
| L40S | `E003` | Planned; `4gpu-pilot`, 1 trial | Slurm job 1773321 obtained four concurrent L40S GPUs and verified that Torch saw all four devices. E003 is the canonical four-rank NCCL release gate. |
| L40S | `E034`, `E037`, `E040`, `E043`, `E046`, `E048`, `E050`, `E056`, `E058`, `E172` | Planned; `4gpu-core`, 3 trials | The eight-GPU L40S node can satisfy these exact four-GPU scaling, strategy, and PyTorch inference conditions. |
| L40S | `E115-E116`, `E119-E120` | Planned; `4gpu-communication`, 3 trials | Four-GPU PCIe communication studies are released behind the E003 gate and remain platform-specific. |
| L40S | `E148`, `E196` | Planned; `vllm-tp4`, 3 trials | The accepted private vLLM environment can now be exercised with four visible L40S GPUs; run only after E003 and a TP=4 canary pass. |
| L40S | `E208-E209` | Planned; `optional-capacity`, 3 trials | A matched checkpointed one-/two-GPU pair preserves TinyLlama, BF16, sequence length 1024, and global batch 16 without changing the original H100 E044/E045 protocol. |
| H100 | `E006` | Planned; `vllm-pilot`, 1 canonical trial | Slurm job 1777640 passed the H100 diagnostic gate with vLLM import, engine start, a served request, and positive generated-token throughput. |
| H100 | `E130`, `E132-E148`, `E178`, `E180-E196` | Planned; platform-separated `vllm-*` waves, 3 trials | The accepted private environment targets `sm_90`; canonical evidence remains gated by E006, then the one-GPU families, TP=2, and TP=4. |
| L40S/H100 | `E151`, `E199` | Planned; `vllm-fp8`, 3 trials per platform | The adapter selects online vLLM FP8 W8A8 explicitly, requires SM 8.9 or newer, records the quantization contract, and binds GPT-2/TinyLlama to BF16 vLLM controls E130/E178. |
| L40S | `E086`, `E093`, `E231-E232` | Acceptance-gated; `training-fp8`, 3 trials | E086/E093 use Transformer Engine delayed-scaling Hybrid FP8. E231/E232 provide BF16 controls using the same converted modules and workload shape. |

The routed vLLM scope now covers all 39 source conditions independently on each platform: one pilot plus 38
performance conditions. TP=4 remains canary-gated until it has started an engine and served a request with four visible
GPUs. All dedicated vLLM waves must
be submitted with `make submit-vllm-wave`, which binds the immutable bundle to the private vLLM environment. The
shared project `.venv` remains the runtime for non-vLLM waves.

## L40S Full-node Extension

`E215-E230` are L40S-only full-node supplemental conditions, not rows released from the `E001-E200` deferred source matrix.
They preserve all original H100 configurations, including the exact rows that will later use the four-GPU H100 node.
The required order is E215 (eight-rank NCCL and PCIe topology), E216-E220 scaling, E221-E224 DDP/FSDP strategy,
E225-E228 communication, E230 TinyLlama TP=8 vLLM canary, then E229 TP=8 performance. E229 is intentionally
TinyLlama-only: its 32 attention heads are divisible by eight, whereas GPT-2's 12 heads are not. These 16 conditions
produce 44 canonical L40S trials and are always analysed as L40S-only evidence.

`E231-E232` are L40S Transformer Engine BF16 controls for source FP8 treatments E086/E093. They hold module
implementation, model revision, workload shape, seed, and step count constant so the treatment comparison changes
numeric precision rather than precision and backend together.

## Deferred After Audit

| Platform | Count | IDs | Blocking condition | Release requirement |
| --- | ---: | --- | --- | --- |
| L40S/H100 | 1 each | `E007` | CUDA 12.4.1 is now available, but TensorRT-LLM, a built engine, and the pinned build/load/request contract are missing. | Validate a compatible package and immutable engine artefact through E007 on each platform. |
| L40S | 10 | `E089`, `E096`, `E131`, `E152`, `E154-E155`, `E175-E176`, `E179`, `E200` | INT8 artefacts, TensorRT performance adapters, or a classification-compatible DistilBERT serving contract are absent. | Supply the specific backend, calibration/artefact hashes, and compatible adapter required by each row. |
| H100 | 12 | `E086`, `E089`, `E093`, `E096`, `E131`, `E152`, `E154-E155`, `E175-E176`, `E179`, `E200` | The shared blockers apply, and E086/E093 still need H100-specific Transformer Engine acceptance and controls. | Supply missing artefacts/adapters and release E086/E093 only after H100 real-model acceptance. |
| L40S | 4 | `E121-E122`, `E125-E126` | These are H100-labelled halves of controlled hardware comparisons. | Run on H100; use `E123-E124` and `E127-E128` for the L40S halves. |
| H100 | 4 | `E123-E124`, `E127-E128` | These are L40S-labelled halves of controlled hardware comparisons. | Run on L40S; use `E121-E122` and `E125-E126` for the H100 halves. |
| L40S | 2 | `E032-E033` | Exact GPT-2 strong-scaling shapes repeatedly exceeded 44.52 GiB. | Preserve for a platform with sufficient memory; do not relabel smaller shapes as exact evidence. |
| H100 | 1 | `E032` | All three preserved exact trials OOMed: 70.34 GiB allocated, 7.65 GiB free, then a 12.27 GiB allocation failed on the observed 79.19 GiB H100. The 544 MiB unallocated reserve is far below the shortfall. | Preserve the original exact row and failure evidence. Any smaller H100 shape must be a separately registered `capacity_adjusted` probe. |
| H100 | 1 | `E068` | All three exact micro-batch-64 trials reached the same capacity failure: 70.34 GiB allocated, 7.65 GiB free, then a further 12.27 GiB request failed. Only 544 MiB was reserved but unallocated. | Preserve E068 and its failed bundle. Register any smaller H100 micro-batch as a new `capacity_adjusted` condition. |
| H100 | 1 | `E172` | No trial has run or failed. The exact four-GPU condition requests the full H100 node (4 GPUs and 64 CPUs), while the active reservation supplies 2 GPUs and 32 CPUs. | Restore the preserved exact row when a full-node H100 reservation or allocation is available; collect three trials separately. |
| L40S | 1 | `E044` | The exact one-GPU TinyLlama batch-16 shape OOMed identically in all three trials: 43.30 GiB allocated, 592.25 MiB free, then a 1000 MiB allocation failed. | Preserve E044 unchanged for H100; use checkpointed E208/E209 as a separately labelled L40S pair. |
| L40S | 2 | `E067-E068` | Exact GPT-2 micro-batch 32/64 shapes repeatedly exceeded 44.52 GiB. | Run exact rows on H100; use the separately labelled L40S capacity probes. |
| L40S | 4 supplemental | `E203-E204`, `E206-E207` | Capacity probes at micro-batch 28/30/26/27 OOMed; E205 at 25 is the largest passing integer probe. | Retain as measured capacity-limit deferrals. |

## Post-Audit Counts

| Platform | Planned source conditions | Deferred source conditions | Planned supplemental probes | Deferred supplemental probes |
| --- | ---: | ---: | ---: | ---: |
| L40S | 180 | 20 | 28 | 4 |
| H100 | 180 | 20 | 5 | 0 |

Every `E001-E200` source row still routes exactly once per platform. The supplemental `E201-E207` rows remain L40S-only
capacity probes, E208/E209 form an L40S-only checkpointed strong-scaling pair, and E231/E232 are backend-matched
BF16 controls for released L40S FP8 treatments. None is silently substituted for an H100 condition.
