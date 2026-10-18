# Current Project Status

## Inference priority update — 18 October 2026

The next campaign prioritises equal-budget inference layouts rather than additional training breadth. The independent E715–E749 study adds eight-GPU L40S TP8/TP4×DP2/TP2×DP4/TP1×DP8 and four-GPU H100 TP4/TP2×DP2/TP1×DP4 comparisons across two prompt lengths and two concurrency levels. Seven current-runtime serving canaries gate 28 three-trial performance conditions. These are configured, not measured results. Existing training evidence remains part of the dissertation; no historical evidence is replaced.

See [the inference crossover runbook](inference_crossover_runbook.md) for scope, priorities, canary gates and launch commands. The dated snapshots below predate this extension.


This snapshot describes repository capabilities. Archived result files and the generated platform tracker remain authoritative for completed trial coverage; a configured wave is not evidence that it has run.

## Long-context serving extension — 26 September 2026

E247–E249 add fresh one-GPU Qwen2.5-7B BF16 serving conditions at 4,096, 6,144 and 8,192 real-text prompt tokens on both platforms. The separate continuation exists because the original GPT-2/TinyLlama prompt-length curves cannot validly reach those contexts. Its pinned real-text inputs, fixed cache/model context, strict request timestamps, content-fingerprinted maximum-shape acceptance gate and submission wrappers are implemented and locally tested. No historical prompt evidence is replaced. See the [launch and interpretation runbook](long_context_serving_runbook.md).

E250–E255 are an H100-only Qwen2.5-7B BF16 training extension. They are intentionally isolated from the small-model strategy series: E250–E251 are one-trial two-GPU FSDP/ZeRO-3 execution canaries, while E252–E255 provide the controlled two- and four-GPU three-trial strategy evidence after the canary verifier passes. E256–E259 are H100-only GPT-2 capacity probes: E256 retains global batch 64 with accumulation, E257 preserves the repeated default-allocator micro-batch-48 OOM, E258 tests the intermediate micro-batch-40 operating point, and E259 is a separate one-trial expandable-segments diagnostic. None may be pooled with the original exact E032/E068 series. See the [sharding study runbook](qwen7b_training_sharding_runbook.md).

The study isolates one-request prefill/latency effects, not long-context batching, concurrent arrivals, HTTP traffic, RAG retrieval or agent behaviour. Its Qwen points must not be pooled with prior small-model prompt sweeps or the separate Qwen low-bit deployment-artifact study.

## Low-bit extension — 25 September 2026

E239–E246 add fresh one-GPU Qwen2.5-7B BF16, AWQ, GPTQ and optional bitsandbytes NF4 conditions on both platforms. The adapter, real-text inputs, content-fingerprinted acceptance gate, submission wrappers and controlled analysis summary are implemented and locally tested. The H100 acceptance and canonical trials are collected; every platform remains a separate evidence stream. No historical raw evidence or original E001–E238 config is replaced. See the [launch and interpretation runbook](lowbit_serving_runbook.md).

This addresses the deployment-format gap once validated and run. It does not yet address larger-model sharded training, a repaired scaling baseline, consumer/non-NVIDIA GPUs or multi-node generalization.

## Ready Now

| Area | Status |
| --- | --- |
| Experiment matrix | 200 source matrix configs, 9 L40S capacity-adjusted conditions, 4 H100 capacity-adjusted conditions, 5 cross-platform telemetry revalidation anchors, 16 L40S full-node conditions, 4 backend-matched FP8 controls, 4 L40S hardware-energy anchors, 8 controlled low-bit inference conditions, 3 controlled long-context serving conditions, and 6 H100-only Qwen2.5-7B sharding conditions are present and load successfully (259 total). |
| Validation | `python -m hpc_ai_perf.cli validate --skip-results` passes with 0 config issues. |
| Local smoke runner | CPU-only synthetic workload is ready and covered by tests. |
| Semantic launch gate | `validate --research-ready` blocks unsupported precision/backend/framework combinations before Slurm rendering or execution. |
| Result pipeline | Immutable, repetition-aware matrix trials, verified config/provenance fields, aggregation, 13 objective-specific analysis tables (including research-question coverage and traceable RQ1/RQ2 findings), GPU telemetry provenance/outlier auditing, progress tracking, and readiness reporting are wired into `python -m hpc_ai_perf.cli refresh`. |
| Distributed execution | One-process-per-GPU `python -m torch.distributed.run`, DDP, FSDP, DeepSpeed ZeRO, rank binding, global work accounting, and rank-safe result writing are implemented. |
| Profiling | Timestamped, allocation-bound nvidia-smi telemetry now separates the post-warm-up measured region from warm-up and profiler finalization. Historical samples remain immutable and are labelled by mapping and energy scope; only sufficient measured-region energy may feed tokens/kWh. |
| Slurm support | Semantic-gated single/bulk rendering, H100/L40S profile matching, CPU allocation checks, and three transformer smoke presets are available. |
| Platform execution | All 200 source rows route exactly once per platform. There are 180 planned L40S source conditions, 43 planned L40S supplemental conditions, 183 planned H100 source conditions, five planned H100 telemetry anchors, and two H100-only Transformer Engine BF16 controls. The 16 L40S-only full-node conditions require E215's eight-rank NCCL/topology gate before scaling, strategy, communication, or TP=8 results are collected. E235-E238 provide immutable measured-region energy anchors for L40S E123/E124/E127/E128 without replacing their original performance evidence. The private PyTorch 2.8.0/CUDA 12.4 stack has passed E006 acceptance on both L40S and H100; each platform routes 39 source vLLM conditions, including E151/E199 online FP8 W8A8, plus three gated Qwen long-context conditions (E247-E249). Transformer Engine FP8 is acceptance-gated on both platforms: L40S E086/E093 use E231/E232, while H100 E086/E093 use E233/E234. H100 E172 is restored to Wave `5` for an unchanged four-GPU/64-CPU recovery; exact H100 E032 and E068 capacity failures, known L40S OOM shapes, TensorRT-LLM, INT8, and classification-adapter rows remain deferred. |
| Evidence retention | L40S/H100 JSON is imported non-destructively into separate roots, scanned for sensitive keys, validated, and covered by tracked SHA-256 manifests. |
| Dashboard | Static dashboard reads processed results and objective-specific analysis summaries. It leads with seven evidence-derived conclusions that state effect sizes, decisions, changed variables, controls, limitations, and traceable source conditions before showing supporting charts. Low-value cross-study rankings, operational submission readiness, and duplicate phase inventories have been removed from the presentation; detailed trial evidence remains in a collapsed audit panel. Its research-evidence section retains RQ1–RQ7 coverage, a compact colour-coded scaling chart with all/strong/weak selectors that preserves all 13 isolated series, and matched throughput-memory findings. The hardware view expands the four dedicated RQ6 rows into 164 exact campaign-wide H100/L40S protocol pairs, with filters, study-level ratio ranges, lead counts, strongest contrasts, and metric drill-downs. Training, precision, memory, hardware, communication, inference, and measured-region energy views remain controlled and unit-separated. |
| Quality gates | `pytest`, `ruff`, and `pre-commit` are configured and passing. |

## Smoke-Ready Workloads

| Workload | Current execution support | Smoke config | Notes |
| --- | --- | --- | --- |
| `synthetic` | Ready | `configs/local/synthetic_smoke.json` | Runs without GPU or optional ML dependencies. |
| `gpt2_small` | Training plus PyTorch/vLLM inference | `configs/local/gpt2_small_smoke.json` | Matrix GPU runs require the matching HPC extras and CUDA. |
| `distilbert` | Training, PyTorch classification inference, concurrent CUDA streams, and multi-GPU data parallelism | `configs/local/distilbert_smoke.json` | Causal-generation adapters are deliberately rejected for this classifier. |
| `tinyllama` | Training plus PyTorch/vLLM inference | `configs/local/tinyllama_inference_smoke.json` | Full-size training remains optional because of GPU-hour cost. |

## Current Source Matrix Breakdown

| Category | Count |
| --- | ---: |
| Source matrix configs | 200 |
| L40S supplemental capacity probes | 9 |
| H100 supplemental capacity probes | 2 |
| Cross-platform telemetry revalidation anchors | 5 |
| L40S full-node extension | 16 |
| Backend-matched FP8 controls | 4 |
| L40S hardware-energy revalidation anchors | 4 |
| Controlled low-bit inference extension | 8 |
| Controlled long-context serving extension | 3 |
| Must Have | 116 |
| Should Have | 71 |
| Optional | 13 |
| Training configs | 107 |
| Inference configs | 93 |

The priority and training/inference counts above describe the original 200 source configs only. The eight low-bit and three long-context conditions are supplemental inference trials; their independent `lowbit-serving`/`lowbit-bnb` and `vllm-long-context` waves add planned conditions to each platform without changing the earlier policy.

## Current HPC State

Both L40S and H100 are available execution targets. They remain separate evidence streams: an L40S result is never
relabelled as H100 evidence. Follow [the platform execution runbook](platform_execution_runbook.md) and create an
immutable submission bundle for each platform/wave. The bundle isolates queued work from later result imports and
commits in the primary checkout.

## Before Each L40S Wave

1. Install the workload extra on the cluster:

   ```bash
   source slurm/modules_for_hpc_ai
   DS_BUILD_OPS=0 DS_BUILD_AIO=0 python -m pip install \
     --constraint slurm/constraints/lovelace_py311_cuda124.txt \
     --upgrade-strategy only-if-needed \
     -e ".[dev,yaml,workloads,distributed]"
   ```

   Do not install a generic vLLM wheel into this environment. The isolated vLLM `0.10.2` environment has passed the
   L40S and H100 acceptance gates and is activated only by the dedicated `vllm-*` waves. Keep the core `.venv` for ordinary
   PyTorch waves; see the [vLLM environment runbook](vllm_environment_runbook.md).

2. Validate the active profile and submit the selected immutable platform wave:

   ```bash
   python -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
   python -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --expected-torch-version 2.8.0 --expected-torch-cuda 12.4 --require-deepspeed
   make check-slurm-profile SLURM_PROFILE=slurm/profiles/cluster_l40s.json
   make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   ```

   Run `check-runtime` from an allocated GPU node. The DeepSpeed portion must pass import/accelerator checks, and
   `make check-cuda-toolchain PLATFORM=l40s EXPECTED_GPUS=1` must pass before CUDA source builds. The active wave 0
   scope is E001, E002, E004, and E005. E003 now runs separately through the one-trial `4gpu-pilot` wave so existing
   wave 0 history remains immutable. vLLM is not a precondition for either ordinary PyTorch gate.

3. Monitor validated trial coverage. The bundle location and Slurm status record are saved automatically under
   ignored `build/platform_waves/`:

   ```bash
   make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   ```

4. After the status command reports all required trials complete, collect and stage verified result evidence:

   ```bash
   make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   git commit -m "data: add completed L40S wave 0 trials"
   ```

5. Review:

   - `results/processed/readiness_report.md`
   - `results/processed/experiment_tracker.md`
   - dashboard at `http://localhost:8000/dashboard/`

## Main Remaining Work

| Area | Remaining work |
| --- | --- |
| HPC verification | Existing one-/two-GPU L40S performance waves are complete. Collect and verify `hardware-energy-revalidation` to fill measured-region energy for E123/E124/E127/E128. Run `4gpu-pilot` before four-GPU work and `8gpu-pilot` before full-node scaling, strategy, communication, or TP=8 work. Local tests cannot validate CUDA/NCCL/Slurm behaviour. |
| Independent trials | Baseline scripts render as three-task Slurm arrays; execute them on Lovelace and inspect confidence intervals and variance from preserved trial records. |
| Backend-dependent rows | E151/E199 use the explicit `vllm-fp8` online-W8A8 path after same-platform BF16 controls E130/E178. Transformer Engine training FP8 E086/E093 is real-model-gated on both platforms and compares only with same-platform controls: E231/E232 on L40S and E233/E234 on H100. TensorRT-LLM E007, DistilBERT FP8 classification, and INT8 rows still need their documented prerequisites. |
| Invalid DistilBERT framework rows | `E154-E155` must be removed from executable scope or redesigned around classification-compatible backends. |
| vLLM tensor parallelism | Two-GPU `E147` and `E195` are complete through `vllm-tp2`. Four-GPU `E148` and `E196` use `vllm-tp4`; supplemental TinyLlama `E229` uses TP=8 only after the one-trial E230 canary passes. GPT-2 is deliberately not given TP=8 because its 12 attention heads are not divisible by eight. These rows measure serving performance, not unavailable request-path NCCL counters. |
| Communication depth | The NCCL microbenchmark and PyTorch trace do not yet provide NVLink counters or full communication/compute attribution. |
| Quality scope | Synthetic held-out checks detect instability; final RQ3 quality claims still need a preregistered real evaluation protocol. |
| Dashboard | Extend objective-specific conclusions as new evidence is collected; trial confidence intervals, explicit insufficient-data states, and conclusion-to-trial traceability are already connected. |

## Deferred Research Scope

Run `python -m hpc_ai_perf.cli validate --skip-results --research-ready` for the authoritative gate; it must pass with zero blockers. The platform tracker lists deferred rows that are intentionally excluded from generated deployment waves. E006 acceptance has passed on both L40S and H100, and both platforms have dedicated one-, two-, and four-GPU vLLM waves; each platform still requires canonical E006 evidence before its performance waves, and TP=4 remains canary-gated. The L40S-only E215-E230 extension does not alter source E001-E200 or any H100 configuration; E230 must pass before E229 TP=8. E032/E033/E044/E067/E068 remain exact-shape L40S OOM deferrals. H100 E032 and E068 are independently deferred after three repeated exact-shape OOM trials each on the 80 GB device. E203/E204/E206/E207 are also deferred after confirming the upper capacity-adjusted probes do not fit. Use E201-E202/E205 for the L40S GPT-2 micro-batch boundary and the matched checkpointed E208/E209 pair for L40S TinyLlama strong scaling. Original E044/E045 remain unchanged for exact H100 execution. E007 remains deferred because a compatible TensorRT-LLM package, built engine, and pinned load/request contract are missing. Other deferred rows still require their documented FP8, quantization, or adapter prerequisites. See [the deferred experiment audit](deferred_experiment_audit.md) for the complete platform-specific list.
