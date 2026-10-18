# Inference-first crossover study

**Decision, 18 October 2026:** prioritise serving layout, prompt length and concurrency before additional training breadth. Completed training evidence remains available for context. No jobs have been submitted for this study.

## Questions and scope

1. With a fixed node GPU budget, when does a larger TP group beat independent replicas?
2. Does that choice change between short and long prompts, or low and higher concurrency?
3. What is the trade-off between TTFT, decode pace, request tails and useful output per GPU-hour?

This is a controlled dense-model execution study, using pinned TinyLlama as an overhead-sensitive workload. It does not establish the best layout for a model that cannot fit on one GPU, production arrival-rate behaviour, general task quality, or direct NCCL/NVLink attribution. Use the existing Qwen 7B TP study as a subsequent larger-model check, after checking its controls and serving contract; do not pool its measurements with these rows.

| Platform | Equal-budget layouts | GPU budget |
|---|---|---:|
| L40S | TP8; two TP4 replicas; four TP2 replicas; eight TP1 replicas | 8 |
| H100 | TP4; two TP2 replicas; four TP1 replicas | 4 |

Every performance layout runs the full 2×2 grid: prompt length **128/1,280**, concurrent requests **1/8**. Output is exactly **128 tokens per request**, using `ignore_eos=True`; BF16, batch size 1, seed 1337, 30 measured rounds after five warm-up rounds. Model context is fixed at 2,048; prefix caching is off; vLLM GPU memory utilisation is fixed at 0.85; eager execution is used. These are fresh conditions with a new comparison scope, not reruns of old rows.

The low-concurrency point tests how much splitting helps one request while replica groups mostly idle. Eight concurrent requests can exercise every L40S replica. These two concurrency levels can demonstrate a change of preference, but cannot pinpoint an exact crossover threshold. If the ranking changes between endpoints, add a separately identified follow-up at intermediate concurrency (2 and 4), rather than claiming the exact threshold. If no ranking changes, report no crossover within the tested range.

## Matrix and gates

| Platform | Wave | IDs | Trials |
|---|---|---|---:|
| L40S | `inference-crossover-canary` | E715–E718 | 4 single trials |
| H100 | `inference-crossover-canary` | E719–E721 | 3 single trials |
| L40S | `inference-crossover-short` | E722–E729 | 8 conditions × 3 |
| L40S | `inference-crossover-long` | E730–E737 | 8 conditions × 3 |
| H100 | `inference-crossover-short` | E738–E743 | 6 conditions × 3 |
| H100 | `inference-crossover-long` | E744–E749 | 6 conditions × 3 |

Total: **35 conditions, 91 trial slots** (seven canaries plus 84 performance trials). Each platform is an independent evidence stream. All arrays and experiments within each wave are serial, but separately submitted waves can still overlap: finish and collect one wave before starting another.

Canaries use the largest planned prompt/concurrency shape, two measured rounds and two warm-up rounds, one per layout. The performance submitter requires collected, checksum-valid, current-config/current-serving-runtime canaries from the same platform. They must demonstrate clean source, the correct layout/GPU/request/output counts, useful throughput, TTFT/decode timing, memory telemetry and balanced replica routing. The new gate is separate from training canaries. Missing timestamps fail acceptance instead of being presented as measured phase evidence.

The serving runtime fingerprint includes layout/router/transformer/measurement code and this study design. Changing those inputs requires new serving acceptance. Because the older training fingerprint also hashes the shared transformer module, a current training gate may reject older training canaries after these serving changes; this does not invalidate historical training results. Freeze the inference source before acceptance and performance runs.

The router now supports bounded 4-/8-GPU replica layouts and uses non-daemon engine processes, allowing vLLM to create its own TP workers. These software paths pass local tests; actual multi-engine startup, CPU/RAM pressure, CUDA memory and timing on each platform still require canaries.

## Prepare without GPU jobs

```bash
make check-inference-crossover
make check-scaling-deep-dive
PYTHONPATH=src python3 -m hpc_ai_perf.cli validate --skip-results --research-ready
PYTHONPATH=src python3 -m hpc_ai_perf.cli plan-serving-layout \
  --config configs/inference_crossover/E715_l40s_tp1_dp8_p1280_c8_canary.json \
  --visible-devices 0,1,2,3,4,5,6,7
```

The model/tokenizer is already in the existing weekend asset inventory. Cache it on the HPC login node with `make prefetch-h100-weekend-models` and verify using `make verify-h100-weekend-model-cache`. Reuse the certified private vLLM environment on both platforms; confirm that it and the shared cache are visible from compute nodes. Do not download weights or build CUDA packages during the allocation.

Render inspection scripts with `make plan-platform-wave PLATFORM=l40s PLATFORM_WAVE=inference-crossover-canary`; repeat for the short/long waves and `PLATFORM=h100`. Inspection scripts are not immutable bundles and must not be used to bypass acceptance or select the wrong environment.

Regeneration is explicit: `make generate-inference-crossover`. Change the declarative design in `src/hpc_ai_perf/config/inference_crossover.py`, then regenerate; do not edit generated conditions by hand.

## On an allocated platform, later

Commit/sync the reviewed execution inputs first. Set the real reservation and certified environment path. L40S needs an eight-GPU allocation; the four-GPU H100 booking cannot execute L40S rows.

```bash
export SBATCH_RESERVATION='ACTUAL_RESERVATION_NAME'
make submit-inference-crossover-canary PLATFORM=l40s \
  VENV_ACTIVATE=/actual/certified/vllm/environment/bin/activate
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=inference-crossover-canary
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=inference-crossover-canary
make verify-inference-crossover-canaries PLATFORM=l40s

make submit-inference-crossover-wave PLATFORM=l40s \
  INFERENCE_CROSSOVER_WAVE=inference-crossover-short \
  VENV_ACTIVATE=/actual/certified/vllm/environment/bin/activate
# Inspect, finish and collect before proceeding to inference-crossover-long.
```

Repeat with `PLATFORM=h100` for the four-GPU study. Default condition estimates are planning allowances, not measured durations: performance conditions allow 0.5 reserved hours per GPU across their three trials, and canaries 0.1. That gives about **25.2 H100 GPU-hours** or **67.2 L40S GPU-hours** for the complete respective study, including canaries. Recalculate after startup/first-trial timings; those estimates do not establish that all work will fit. Preserve recovery/collection time and stop before the reservation ends. Do not queue both the old E702–E714 campaign and this complete fresh matrix automatically.

## Analysis and dissertation evidence

After collection, `make refresh` adds results to `inference_tp_crossover_summary.csv`. Acceptance rows are excluded. Groups retain comparison scope, model revision, runtime versions, serving fingerprint, input identity and request shape; they do not combine these fixed-output/no-prefix-cache conditions with historical rows. Within each matched group the baseline is the TP1 replica layout, not a single GPU. Report the independent-trial throughput means and t-based 95% intervals; latency percentiles are within-trial statistics. With only 30 requests in concurrency-1 trials, p99 is exploratory, not a production-tail estimate.

Preserve raw request samples and failures. Backend request timestamps do not include HTTP/network or all parent-router admission overhead; interpret them alongside whole-round throughput. vLLM may reserve most of the configured memory budget for cache, so total device memory is not a direct measure of KV occupancy or weight-sharding savings. Record per-GPU memory and topology, throughput, request rate, TTFT/TPOT, and GPU-hours. The generic adapter has no request-path collective attribution; “communication dominates” remains a mechanism hypothesis unless profiling supports it. No latency SLO is invented: goodput remains unclaimed until a justified SLO is explicitly preregistered. The study measures execution performance, not language quality or time to a quality target.

Cross-platform results compare complete hardware platforms. NVLink/PCIe is not the sole changed variable: compute and memory also differ. Within a platform, the fixed-budget layouts isolate the practical choice of splitting versus replication more cleanly.

Local verification: full suite passed (1,160 tests, 21 skipped), and the added analysis-isolation regression passes. Ruff, semantic validation, both matrix design checks and shell syntax checks for all 35 previews pass. New canary gates reject missing evidence on both platforms as expected. These checks do not establish CUDA or multi-engine startup readiness.
