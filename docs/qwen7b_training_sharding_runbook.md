# Qwen2.5-7B H100 Sharding Study

This H100-only extension addresses the study's principal distributed-training external-validity gap: the earlier
strategy rows use models small enough that sharding is often optional. It introduces a separately labelled
Qwen2.5-7B BF16 causal-LM study of FSDP full sharding and DeepSpeed ZeRO-3. Whether sharding is necessary
at this particular load/optimizer shape must be measured with a matching DDP control, not assumed.

It is a new study family (`qwen25_7b_h100_sharding_v1`), not a replacement for GPT-2, DistilBERT, or TinyLlama
evidence. Never pool it with the earlier small-model strategy rankings.

## Controlled design

| Control | Value |
| --- | --- |
| Platform | One H100 node only (`gpu_h100`) |
| Model | `Qwen/Qwen2.5-7B-Instruct` at `a09a35458c702b33eeacc393d103063234e8bc28` |
| Precision and loading | BF16 runtime and explicit BF16 checkpoint loading |
| Data | Deterministic synthetic causal-LM tokens |
| Per-GPU micro-batch | 1 sequence |
| Sequence length | 512 tokens |
| Effective global batch | 16 sequences |
| Optimisation controls | AdamW, learning rate `1e-5`, activation checkpointing enabled |
| FSDP wrapping | One Qwen decoder layer per FSDP unit, `limit_all_gathers=True`, no forward prefetch |
| Telemetry | UUID-mapped `nvidia-smi`, memory trace, and NCCL all-reduce probe |

Within a GPU-count pair, FSDP full-shard and ZeRO-3 share requested workload settings, but their optimizer-state
precision and internal accumulation behavior are not necessarily identical. The two- and four-GPU
conditions preserve the effective global batch by changing accumulation from 8 to 4 steps; treat that as a
controlled scaling sensitivity, not a fixed-accumulation speedup curve.

## Evidence plan

| Stage | Conditions | GPUs | Trials | Purpose |
| --- | --- | ---: | ---: | --- |
| Gate | E250 FSDP, E251 ZeRO-3 | 2 | 1 | Verify the exact model, BF16 load path, activation checkpointing, loss, telemetry, and both runtimes. Not performance evidence. |
| Canonical | E252 FSDP, E253 ZeRO-3 | 2 | 3 each | Two-GPU controlled strategy comparison. |
| Canonical | E254 FSDP, E255 ZeRO-3 | 4 | 3 each | Four-GPU controlled strategy comparison. |

## Run order

On the login node, use the ordinary project environment (not the vLLM or Transformer Engine environment):

```bash
# Run from the repository root.
source slurm/modules_for_hpc_ai
source .venv/bin/activate

make submit-qwen7b-training-canary PLATFORM=h100
make status-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training-canary
```

After both one-trial canaries are completed and provenance-valid, collect them into immutable evidence and certify
their metrics before scheduling the three-trial wave:

```bash
make collect-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training-canary
make stage-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training-canary
git diff --cached --stat
git commit -m "data: add H100 Qwen 7B sharding canary trials"

make verify-qwen7b-training-canary PLATFORM=h100
make submit-qwen7b-training-wave PLATFORM=h100
```

The canonical wave contains both two- and four-GPU jobs. Run it only while all four H100 GPUs are available, then:

```bash
make status-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training
make collect-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training
make stage-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training
git diff --cached --stat
git commit -m "data: add H100 Qwen 7B sharding study trials"
```

## Collecting a completed canary while its paired canary is repaired

If one canary completes and the other fails, preserve the completed immutable result immediately rather than rerunning
it. For example, if E251 completed while E250 needs a runtime repair:

```bash
export BUNDLE=../hpc-ai-submissions/qwen7b-canary-bundle
make import-result-evidence SOURCE="$BUNDLE/results/E251"
make validate-result-evidence
git add results/raw/h100/E251 results/raw/h100/SHA256SUMS
git commit -m "data: add H100 Qwen 7B ZeRO-3 canary trial"
```

After pulling the runtime repair, use recovery for the missing condition only:

```bash
make retry-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-training-canary
```

The retry logic recognises E251's collected evidence and schedules only E250. Once E250 completes, run the normal
wave collection command; the canary verifier requires evidence for both conditions.

## Failure handling

Do not retry a deterministic OOM or silently lower the sequence length, model size, or effective batch. Preserve the
failed immutable result. The initial E250 canary showed that root-only FSDP wrapping attempts a model-scale all-gather;
the runtime now uses decoder-layer wrapping before it is retried. If that layer-wise path still reaches a capacity limit,
add a new H100-only `capacity_adjusted` condition with an explicit objective and do not combine it with E252–E255. If
only the four-GPU jobs remain unschedulable, collect the completed two-GPU evidence and record the four-GPU allocation
constraint rather than treating it as a model or strategy failure.

## Fresh update controls (E261–E269)

The September 26 evidence reports different final losses between native FSDP and ZeRO-3. This is not proof of
a defect, but prevents assuming numerically equivalent optimization. Preserve E250–E255 unchanged.
The new `qwen25_7b_h100_update_v2` family adds DDP feasibility and records initial loss, backward/update
counts, engine sample accounting, and actual Adam-state dtypes. ZeRO-3 uses explicit accumulation boundaries,
configured GAS 8 (two GPUs) or 4 (four GPUs), and disables DeepSpeed's second loss division because the common
loop already scales the loss. Legacy configurations keep the old external-accumulation behavior.

| Wave | IDs | GPUs | Repetitions |
| --- | --- | ---: | ---: |
| `qwen7b-update-canary` | E262 FSDP, E263 ZeRO-3 | 2 | 1 each |
| `qwen7b-update-controls-2gpu` | E265 FSDP, E266 ZeRO-3 | 2 | 3 each |
| `qwen7b-update-controls-4gpu` | E268 FSDP, E269 ZeRO-3 | 4 | 3 each |

E261's two-GPU DDP canary failed in job 1781885, bundle
`h100-wave-qwen7b-update-canary-2537fe9f-20260926t184439z`. PyTorch reported 77.41 GiB allocated,
79.07 GiB used by the process, and a failed 130 MiB allocation with 115.50 MiB free. E261 and its
matching three-trial DDP condition E264 are now deferred on H100. E267 is separately deferred because
four-GPU DDP has no independent successful feasibility gate; the two-GPU failure does not establish
four-GPU infeasibility. All three source configurations remain available for a future separately
validated investigation. This finding describes the tested deployment, not every possible DDP or
allocator configuration.

Collected E262/E263 each recorded six optimizer updates and 48 backward calls. Their initial losses
were 14.00871 and 14.01416 (a 0.039% difference). These current-contract canaries certify the sharded
update waves, which now contain six trials each. Their one-trial timings are diagnostic; the new
three-trial conditions supply the performance evidence. Native optimizer states were BF16 and
DeepSpeed states FP32, so the comparison remains between configured runtime stacks.

```bash
# Submit canaries only if current E262/E263 evidence has not already been collected.
# make submit-qwen7b-update-canary PLATFORM=h100
make status-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-update-canary
make collect-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-update-canary
make verify-qwen7b-update-canary PLATFORM=h100
make stage-platform-wave PLATFORM=h100 PLATFORM_WAVE=qwen7b-update-canary
git diff --cached --stat
git commit -m "data: add H100 Qwen training update canaries"
make submit-qwen7b-update-wave PLATFORM=h100 QWEN7B_UPDATE_WAVE=qwen7b-update-controls-4gpu PLATFORM_TIME_LIMIT=00:15:00
make submit-qwen7b-update-wave PLATFORM=h100 QWEN7B_UPDATE_WAVE=qwen7b-update-controls-2gpu PLATFORM_TIME_LIMIT=00:15:00
```

The gate requires both current FSDP/ZeRO-3 canaries, contract/content hashes (not HEAD), update/sample
counts, finite losses, and initial loss agreement within 5%. It also requires the unvalidated DDP
conditions to remain deferred, including when submission selects an alternative execution matrix.
It does **not** claim matching final loss or optimizer precision. Retain E261's failed bundle as
feasibility evidence for the report. After the explicit policy deferral, normal canary collection can
collect the two successful siblings without retrying E261. Native BF16 Adam states versus DeepSpeed
FP32 states must remain visible in conclusions.
