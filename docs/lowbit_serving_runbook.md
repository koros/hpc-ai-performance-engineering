# Controlled low-bit inference extension

Status: v1 has collected H100 evidence; the new v2 quality extension is implemented and locally tested but
has not yet passed its GPU gate or produced canonical trials.
The protocol is `qwen25_7b_lowbit_v1`. Existing E001–E238 configs and raw evidence are unchanged.

## Broader quality study: H100 E270–E273

`lowbit-quality` uses the separate `qwen25_7b_lowbit_quality_v2` protocol. Its BF16/AWQ/GPTQ/NF4 arms each
score 128 non-overlapping 512-token WikiText test windows (65,408 predicted tokens), versus 32 × 256 in v1.
E270 is a fresh BF16 control; E271–E273 never borrow E239's shorter quality evaluation. Publisher artifacts,
BF16 compute, request batch 1, 512-token performance prompts, 128 generated tokens, fixed KV cache and the
preregistered 10% relative perplexity tolerance stay constant. This measures broader text likelihood, not
instruction/task accuracy or a pure quantizer effect. Repetitions reuse the same held-out windows; they do
not multiply the number of distinct quality examples.

```bash
make prefetch-lowbit-assets LOWBIT_WAVE=lowbit-quality
make submit-lowbit-acceptance PLATFORM=h100 LOWBIT_WAVE=lowbit-quality
# Wait for the printed diagnostic job ID to finish, then replace <job-id> below.
make verify-lowbit-acceptance PLATFORM=h100 LOWBIT_WAVE=lowbit-quality LOWBIT_ACCEPTANCE_JOB=<job-id>
make submit-lowbit-wave PLATFORM=h100 LOWBIT_WAVE=lowbit-quality LOWBIT_ACCEPTANCE_JOB=<job-id>
make status-platform-wave PLATFORM=h100 PLATFORM_WAVE=lowbit-quality
make collect-platform-wave PLATFORM=h100 PLATFORM_WAVE=lowbit-quality
```

This gate automatically checks all four methods, including NF4, and the full new evaluation shape. Do not
reuse an older acceptance result. Runtime changes invalidate the earlier gate even when package versions
match. The generated low-bit summary carries `study_protocol` and uses independent controls for v1/v2.

## Research purpose and comparison boundary

This extension addresses RQ3 (precision and quality), RQ5 (useful work and GPU energy), and RQ6
(H100/L40S comparison) with a larger inference model and explicit deployment methods. It does not
reuse GPT-2/TinyLlama or FP8 results as controls, and does not relabel the previously deferred INT8 rows.

The model is [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), with its publisher's
[AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ) and
[GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4) releases. The optional
bitsandbytes arm quantizes the pinned BF16 checkpoint in flight to **NF4**, not integer INT4 or INT8.
vLLM's [quantization documentation](https://docs.vllm.ai/en/v0.10.2/features/quantization/) describes
the supported backends; backend support is not evidence of community market share.

The question is: **which released artifact/backend combination gives a useful speed, tail-latency,
memory and energy trade-off at a fixed load, without exceeding a declared text-quality tolerance?**
AWQ/GPTQ upstream calibration and their exact pre-quantization base revisions are not fully disclosed.
Consequently this is a *publisher-artifact deployment comparison*, not an isolated causal comparison
of quantizer algorithms trained on identical calibration data. Calibration overlap with evaluation
text is unknown. Do not claim that any method is universally best for open-source users.

## New matrix

Every condition uses one GPU and three independent process runs on each platform.

| Method | Batch 1 | Batch 8 | Wave | Weight representation / backend |
| --- | --- | --- | --- | --- |
| Fresh BF16 control | E239 | E240 | `lowbit-serving` | BF16 / vLLM |
| Publisher AWQ | E241 | E242 | `lowbit-serving` | INT4, group 128 / AWQ Marlin |
| Publisher GPTQ | E243 | E244 | `lowbit-serving` | INT4, group 128 / GPTQ Marlin |
| In-flight bitsandbytes | E245 | E246 | `lowbit-bnb` | NF4 / bitsandbytes |

Core: 18 trials per platform, 36 across both. Optional NF4: six per platform, 12 across both.
H100's four-GPU node and L40S's eight-GPU node are **not** required in full for these waves.
Canonical profiles allocate 16 CPUs on H100 and eight on L40S; acceptance allocates eight on either.
Within-platform comparisons hold the CPU allocation constant. Cross-platform results describe the
configured platform stacks, not GPU-only causality with identical host resources.

`exact` means the new protocol is reproduced without a capacity-adjusted workload shape. It does not
mean the quantized artifacts have identical weights or that these trials can be pooled with old studies.

## What changes and what stays fixed

The changing variables are method, batch size (1 or 8), and platform. Only compare methods at the
same batch size and platform first; then compare matching conditions across platforms.

| Control | Protocol |
| --- | --- |
| Artifact/tokenizer | Immutable Hugging Face revisions in `src/hpc_ai_perf/config/lowbit.py`; shared base tokenizer |
| Engine | Private vLLM 0.10.2, V0 eager, tensor parallelism 1; no prefix caching, chunked prefill or CPU offload |
| Compute / KV cache | BF16 compute; BF16 KV cache, 1,024 blocks × 16 tokens, fixed across methods |
| Request shape | 512 prompt tokens, exactly 128 generated tokens, greedy decoding, EOS ignored |
| Performance text | 64 deterministic WikiText-2 raw validation windows, reused cyclically |
| Warm-up / timing | Three batches of warm-up; at least two measured batches and 30 measured seconds |
| Safety cap | 300 measured batches; fail if they do not cover 30 seconds rather than accepting sparse sampling |
| Quality text | 32 separate WikiText-2 raw test windows × 256 tokens; 8,160 scored next-token targets |
| Quality tolerance | At most 10% relative perplexity increase against the fresh matching BF16 control |
| Repetition | Three fresh process runs, seed 1337; measures system variability, not training-seed uncertainty |
| Telemetry | 0.1-second requested GPU sampling interval, measured-region energy, verified GPU attribution |

WikiText rows are short paragraphs. The adapter concatenates nonempty tokenized rows with EOS
separators and forms non-overlapping fixed-length windows. It records token hashes for both splits.
This is real-corpus text, but not a conversational traffic trace. No chat template is applied.

Model loading, warm-up and quality evaluation occur **outside** the throughput/energy interval.
Loading and in-flight NF4 quantization costs are therefore not steady-state serving costs; this version
does not measure cold-start performance. Requests are submitted as closed offline batches, not through
an HTTP server or an open-loop arrival process. TTFT and p95 use actual vLLM request timestamps; missing
timestamps fail the run instead of substituting batch wall time. Tail percentiles from a short pilot
are descriptive, not a production service-level guarantee.

Quality is token negative log-likelihood/perplexity on a small preregistered text sample, not instruction
following, task accuracy, convergence, or a full benchmark score. A result within tolerance is only
"within the text-perplexity tolerance on this sample." Three timing repetitions of the same text do not
provide independent estimates of quality generalization.

Memory is measured in the actual vLLM worker. Report model-loader memory separately from peak CUDA
allocated/reserved memory, which includes the **fixed KV cache**. Do not interpret the latter as pure
weight size. GPU energy excludes host CPUs and whole-node power. Eligible energy must have at least
10 interior observations and no sampling gap over one second, as well as valid device mapping and
measured-region coverage. Interpolation between only two distant samples is not sufficient here.

## Prepare before the GPU window

Run from the project root. The project `.venv` drives the Make/CLI lifecycle; the low-bit jobs activate
the private vLLM environment themselves. `LOWBIT_ENV` defaults to `VLLM_ENV`, normally
`$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo`.

```bash
source slurm/modules_for_hpc_ai
source .venv/bin/activate
export LOWBIT_ENV="$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo"
```

The private environment needs its already-validated Torch/CUDA/Gloo/NCCL and vLLM 0.10.2 stack, plus
`transformers`, `tokenizers`, `datasets`, and `huggingface-hub`. The optional NF4 arm requires
`bitsandbytes==0.47.0`. Do not upgrade Torch/vLLM or install generic binary replacements over the cluster
stack. Diagnose missing dependencies against the private interpreter, not the project `.venv`.
Install/resolve any missing packages before acceptance; keep the resulting environment unchanged
while jobs are queued or running. There is no automatic runtime installer in this extension.

If NF4 is intended, provision bitsandbytes **before** running even the BF16 controls. Installed package
identities are part of the comparison contract. Adding it later changes that contract and can require
fresh controls as well as fresh acceptance. The optional arm is split so it need not block AWQ/GPTQ.

Prefetch the three pinned model snapshots, shared tokenizer and dataset before spending the reservation:

```bash
make prefetch-lowbit-assets
```

This downloads several large weight files; check cache quota and allow ample disk headroom. The HF cache
must be visible from compute nodes. For a smaller check without model weights:

```bash
PYTHONPATH="$PWD/src" "$LOWBIT_ENV/bin/python" \
  scripts/prefetch_lowbit_assets.py --metadata-only
```

Metadata-only validation, including both real-data token streams, has passed locally. It does **not**
validate GPU kernels, the full weight downloads or available memory on Lovelace.

## Acceptance, then canonical trials

Start on one platform. Substitute `h100` throughout for the independent H100 gate and wave.

```bash
make submit-lowbit-acceptance PLATFORM=l40s
# If bitsandbytes has already been provisioned, use this INSTEAD:
# make submit-lowbit-acceptance PLATFORM=l40s LOWBIT_INCLUDE_BNB=1
```

Note the submitted job ID. Check its scheduler status and `slurm/lowbit-acceptance-JOB_ID.out/.err`;
after it finishes, set the variable to that actual ID (replace the placeholder):

```bash
export LOWBIT_ACCEPTANCE_JOB=JOB_ID
make verify-lowbit-acceptance PLATFORM=l40s
make submit-lowbit-wave PLATFORM=l40s
```

Acceptance runs batch 8 through BF16, AWQ and GPTQ (optionally NF4), in separate fresh processes.
It validates successful generation, finite quality, expected quantized worker modules, fixed cache,
real per-request timings, measured duration and energy coverage. NF4 additionally requires observed
NF4 quantization state. All eight canonical configs are fingerprinted, so a later batch-1 change also
invalidates the gate. Accepting one platform does not certify the other.

The certificate is `build/lowbit_acceptance/<platform>/<job-id>/`; it is diagnostic evidence and must
**not** be imported into `results/raw`. Source-code/config content and dependency identities invalidate
stale acceptance; a documentation-only git commit does not. On failure, inspect the diagnostic JSON and
Slurm error log before retrying. Do not bypass the gate by submitting rendered scripts directly.

Follow the normal immutable lifecycle after submission:

```bash
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=lowbit-serving
# Only after the required trials are complete and provenance-valid:
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=lowbit-serving
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=lowbit-serving
git diff --cached --stat
git commit -m "data: add L40S controlled BF16 AWQ GPTQ trials"
```

For failed/incomplete trials, fix the diagnosed issue and use
`make retry-lowbit-wave PLATFORM=l40s` with the appropriate `LOWBIT_ACCEPTANCE_JOB`. A runtime/config
fix may require a new acceptance job. Preserve previous evidence; do not edit a queued bundle or raw JSON.

After the core controls exist, the optional arm uses its stricter gate:

```bash
make verify-lowbit-acceptance PLATFORM=l40s LOWBIT_WAVE=lowbit-bnb
make submit-lowbit-wave PLATFORM=l40s LOWBIT_WAVE=lowbit-bnb
# Use PLATFORM_WAVE=lowbit-bnb for status / collect / stage.
```

The selected diagnostic directory must include NF4 from an acceptance run with `LOWBIT_INCLUDE_BNB=1`.
Do not submit twice to check status. Each core wave is six three-task job arrays; each optional wave is
two. Scheduler completion alone is not proof of valid workload results.

## Analysis outputs

After collecting canonical evidence, the normal `make refresh` pipeline generates
`results/analysis/lowbit_summary.csv` when low-bit results are present. It provides:

- throughput mean and Student-t 95% interval for independent process runs;
- throughput and p95 changes against the matching BF16 control;
- model-memory savings and worker peak allocated memory;
- perplexity change and the preregistered quality-tolerance result;
- eligible GPU joules per generated token and relative energy change;
- evidence status and source paths for audit.

Missing controls, incomplete runs, protocol drift or ambiguous repeated-trial identities suppress comparative
conclusions. Bad energy coverage suppresses energy comparisons without discarding valid performance data.
Methods, batch sizes and platforms are not pooled. Low-bit rows are excluded from the generic precision
average; their dedicated summary preserves the deployment-method distinction. This change does not add a
new dashboard panel or publish the website. Review the dedicated CSV before deriving recommendations.

## Remaining research extensions

This is the first high-impact implementation slice, not completion of every external-validity recommendation.
Next priorities are a fresh feasible one-GPU baseline for a complete controlled scaling series, and a separately
designed larger-model training study where FSDP/ZeRO sharding is necessary. The 7B inference results cannot
answer that training question. Real instruction-quality evaluation and longer-context traffic are also still
outside this pilot. Consumer/non-NVIDIA GPUs and multi-node execution remain explicit scope boundaries.
There are four H100 GPUs in the available node: do not schedule an eight-H100 study on that hardware.
