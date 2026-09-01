# Deferred Experiment Audit

Audit date: 2026-09-01.

The L40S private PyTorch 2.8.0/CUDA 12.4 build now provides both Gloo and NCCL, the pinned vLLM 0.10.2 environment
loads successfully, and the L40S E006 acceptance gate completed an engine start and served request. This resolves the
former L40S-wide vLLM environment blocker. It does not establish H100 acceptance or remove independent hardware,
memory-capacity, TensorRT-LLM, FP8, quantization, or adapter blockers.

## Released From Deferred State

| Platform | IDs | New execution state | Reason |
| --- | --- | --- | --- |
| L40S | `E006` | Planned; `vllm-pilot`, 1 canonical trial | The diagnostic acceptance gate passed; the immutable pilot is still required as canonical matrix evidence. |
| L40S | `E130`, `E178` | Planned; `vllm-framework`, 3 trials | One-GPU GPT-2 and TinyLlama vLLM framework baselines now use the accepted private environment. |
| L40S | `E132-E137`, `E180-E185` | Planned; `vllm-batching`, 3 trials | One-GPU batching studies are supported by the accepted adapter/runtime. |
| L40S | `E138-E142`, `E186-E190` | Planned; `vllm-concurrency`, 3 trials | One-GPU concurrency studies are supported by the accepted adapter/runtime. |
| L40S | `E143-E146`, `E191-E194` | Planned; `vllm-prompt`, 3 trials | One-GPU prompt-length studies are supported by the accepted adapter/runtime. |
| L40S | `E147`, `E195` | Planned; `vllm-tp2`, 3 trials | The L40S node exposes the two GPUs required by these tensor-parallel rows. |

This releases 35 L40S source conditions: one pilot plus 34 performance conditions. All dedicated vLLM waves must be
submitted with `make submit-vllm-wave`, which binds the immutable bundle to the private vLLM environment. The shared
project `.venv` remains the runtime for non-vLLM waves.

## Deferred After Audit

| Platform | Count | IDs | Blocking condition | Release requirement |
| --- | ---: | --- | --- | --- |
| H100 | 37 | `E006`, `E130`, `E132-E148`, `E178`, `E180-E196` | The private stack targets `sm_90`, but the serving gate has only passed on L40S. | Run the E006 acceptance job on H100; add H100 vLLM waves only after import, engine start, and request service pass there. |
| L40S/H100 | 1 each | `E007` | CUDA 12.4.1 is now available, but TensorRT-LLM, a built engine, and the pinned build/load/request contract are missing. | Validate a compatible package and immutable engine artefact through E007 on each platform. |
| L40S/H100 | 14 each | `E086`, `E089`, `E093`, `E096`, `E131`, `E151-E152`, `E154-E155`, `E175-E176`, `E179`, `E199-E200` | FP8/quantized artefacts, TensorRT performance adapters, or a classification-compatible DistilBERT serving contract are absent. | Supply the specific backend, calibration/artefact hashes, and compatible adapter required by each row. |
| L40S | 4 | `E121-E122`, `E125-E126` | These are H100-labelled halves of controlled hardware comparisons. | Run on H100; use `E123-E124` and `E127-E128` for the L40S halves. |
| H100 | 4 | `E123-E124`, `E127-E128` | These are L40S-labelled halves of controlled hardware comparisons. | Run on L40S; use `E121-E122` and `E125-E126` for the H100 halves. |
| L40S | 2 | `E032-E033` | Exact GPT-2 strong-scaling shapes repeatedly exceeded 44.52 GiB. | Preserve for a platform with sufficient memory; do not relabel smaller shapes as exact evidence. |
| L40S | 2 | `E067-E068` | Exact GPT-2 micro-batch 32/64 shapes repeatedly exceeded 44.52 GiB. | Run exact rows on H100; use the separately labelled L40S capacity probes. |
| L40S | 4 supplemental | `E203-E204`, `E206-E207` | Capacity probes at micro-batch 28/30/26/27 OOMed; E205 at 25 is the largest passing integer probe. | Retain as measured capacity-limit deferrals. |
| L40S | 17 | `E003`, `E034`, `E037`, `E040`, `E043`, `E046`, `E048`, `E050`, `E056`, `E058`, `E115-E116`, `E119-E120`, `E148`, `E172`, `E196` | Each row requests four GPUs while the L40S node exposes two. | Obtain a verified four-GPU L40S allocation; two-GPU substitution is not valid. |

## Post-Audit Counts

| Platform | Planned source conditions | Deferred source conditions | Planned supplemental probes | Deferred supplemental probes |
| --- | ---: | ---: | ---: | ---: |
| L40S | 160 | 40 | 3 | 4 |
| H100 | 144 | 56 | 0 | 0 |

Every `E001-E200` source row still routes exactly once per platform. The supplemental `E201-E207` rows remain L40S-only
capacity probes and are never pooled with exact H100 evidence.
