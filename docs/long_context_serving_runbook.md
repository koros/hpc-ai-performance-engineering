# Controlled long-context vLLM continuation

Status: implemented and locally validated; platform-specific GPU acceptance and canonical trials are **not yet run**.

The original `vllm-prompt` sweep ends at 896 tokens for GPT-2 and 1,920 tokens for TinyLlama. Those
models have fixed context limits of 1,024 and 2,048 tokens respectively, so adding 4k–8k points to
either series would be invalid. This is a separate, labelled continuation using
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), whose pinned BF16 artifact has
enough context capacity. Do not pool it with the existing GPT-2/TinyLlama prompt-length curves.

## Research question and boundary

The controlled question is: **as prompt context grows from 4,096 to 8,192 real-text tokens, how do
prefill throughput, time-to-first-token, tail latency, worker memory and eligible GPU energy change on
each platform?** It strengthens the realistic-context part of RQ5, RQ6 and RQ7.

This is not a RAG or agent traffic simulation. The protocol submits one offline request at a time, so
it isolates the context-length/prefill curve. It does **not** estimate batching or concurrency behaviour
at long contexts, HTTP queueing, retrieval latency, tool calls, answer quality, or a production SLO.

## Matrix

Each condition is a fresh process run, repeated three times on L40S and independently three times on
H100. The `vllm-long-context` wave uses one GPU per trial.

| Prompt tokens | Experiment | Fixed output | Purpose |
| ---: | --- | ---: | --- |
| 4,096 | E247 | 128 | Lower realistic RAG/agent-context point. |
| 6,144 | E248 | 128 | Midpoint that tests whether prefill cost is linear or changes regime. |
| 8,192 | E249 | 128 | Upper point and same-platform acceptance diagnostic shape. |

The model, performance text and all engine/cache controls are fixed. `exact` means this new protocol
is reproduced as configured on that platform; it does not make it an exact counterpart of the original
small-model prompt rows.

## Controls and measurements

| Item | Fixed protocol |
| --- | --- |
| Model/tokenizer | Qwen2.5-7B-Instruct BF16, immutable revision `a09a35458c702b33eeacc393d103063234e8bc28` |
| Text input | Eight deterministic, non-overlapping WikiText-2 raw validation token windows, immutable dataset revision |
| Request shape | One request, one concurrent request, 128 generated tokens, greedy generation with EOS ignored |
| Engine | Private vLLM 0.10.2, V0 eager, TP=1; prefix caching, chunked prefill and CPU offload disabled |
| Context/cache | `max_model_len=9216`; 1,024 cache blocks × 16 tokens (16,384 token capacity) fixed for all three points |
| Timing | Three warm-ups, at least 30 measured seconds, strict actual per-request timestamps, three fresh-process repetitions |
| Telemetry | Requested 0.1-second GPU sampling, UUID-bound GPU mapping, measured-region energy only |

WikiText is concatenated with EOS separators before token windows are formed, avoiding a synthetic
repeated-token prompt. It is still an offline corpus stream, not a representative multi-turn prompt
distribution. No task-quality score is collected: weights and BF16 compute are held constant, and the
experiment measures serving cost rather than model quality.

The fixed cache is intentional: model-memory and worker-peak memory include its allocation and must
not be reported as pure weight memory. The implementation records both model-loader memory and worker
peak CUDA memory. Energy is GPU-only—not CPU or whole-node energy—and is eligible only with at least
10 interior samples, no gap over one second, valid device attribution and measured-region coverage.

## Prepare before a GPU reservation

Run from the repository root. The project `.venv` controls the Make/CLI lifecycle; jobs activate the
private Gloo-enabled vLLM environment. Do not install or upgrade packages during a queued wave.

```bash
source slurm/modules_for_hpc_ai
source .venv/bin/activate
export LONG_CONTEXT_ENV="$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo"
make prefetch-long-context-assets
```

The prefetch command downloads the pinned BF16 Qwen checkpoint, tokenizer and dataset. Verify that the
Hugging Face cache is visible from compute nodes and has sufficient quota before a reservation. For a
quick contract/data check without downloading model weights:

```bash
PYTHONPATH="$PWD/src" "$LONG_CONTEXT_ENV/bin/python" \
  scripts/prefetch_long_context_assets.py --metadata-only
```

Metadata-only validation confirms that eight deterministic windows are available at every target length;
it does not validate GPU memory, vLLM kernels or full model download availability.

## Per-platform acceptance, then canonical trials

The maximum 8,192-token shape must pass on the target GPU before canonical work is submitted. An E006
vLLM import/engine pass is necessary but not enough: it does not prove long-context prefill, cache
allocation, strict timestamping or telemetry coverage.

Start with L40S; substitute `h100` for a separate H100 acceptance and wave:

```bash
make submit-long-context-acceptance PLATFORM=l40s
# Record the Slurm job ID, wait for completion, then:
export LONG_CONTEXT_ACCEPTANCE_JOB=JOB_ID
make verify-long-context-acceptance PLATFORM=l40s
make submit-long-context-wave PLATFORM=l40s
```

The acceptance run writes a diagnostic file below
`build/long_context_acceptance/<platform>/<job-id>/`. It must show a completed E249-equivalent request,
the expected GPU, unquantized BF16 Qwen, the pinned artifact/config/runtime fingerprints, fixed cache,
positive throughput and latency, a 30-second interval, actual request timestamps and eligible measured
region energy. It is diagnostic evidence only: **do not import it into `results/raw`**.

Acceptance is content- and dependency-bound. Changes to the model/config/workload/runtime or private
environment require a new target-platform acceptance. Documentation-only changes do not. Do not bypass
the gate by using generic `submit-platform-wave` directly.

After submission, follow the normal immutable-evidence lifecycle:

```bash
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-long-context
# Only once all three trials for E247–E249 are complete and provenance-valid:
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-long-context
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=vllm-long-context
git diff --cached --stat
git commit -m "data: add L40S Qwen long-context vLLM trials"
```

For a failed condition, keep its bundle and JSON evidence, diagnose the specific failure, repair the
root cause, run a fresh acceptance if the contract/runtime changed, then use:

```bash
make retry-long-context-wave PLATFORM=l40s
```

Never resubmit merely to check scheduler status, modify a queued bundle, or overwrite raw evidence.

## Analysis and reporting

After collection, run `make refresh`. The standard inference, phase, trial and hardware summaries will
include these evidence records. Analyse them only as three platform-isolated points in the Qwen BF16
long-context curve. Report the changing variable (prompt tokens) and fixed controls in the caption.

Useful comparisons are 4,096 → 6,144 → 8,192 tokens within a platform, then matching prompt lengths
across platforms. Do not average them with the existing short-prompt model families, FP8 rows, AWQ/GPTQ
rows, batch/concurrency sweeps, or capacity-adjusted trials. Missing or ineligible energy telemetry
suppresses energy claims but does not invalidate measured throughput/latency evidence.
