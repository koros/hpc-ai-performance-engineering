# Qwen2.5-7B H100 Sharding Study

This H100-only extension addresses the study's principal distributed-training external-validity gap: the earlier
strategy rows use models small enough that sharding is often optional. It introduces a separately labelled
Qwen2.5-7B BF16 causal-LM study in which FSDP full sharding and DeepSpeed ZeRO-3 are necessary, meaningful
alternatives.

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
| Telemetry | UUID-mapped `nvidia-smi`, memory trace, and NCCL all-reduce probe |

Within a GPU-count pair, FSDP full-shard and ZeRO-3 differ only in distributed strategy. The two- and four-GPU
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
cd ./
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

## Failure handling

Do not retry a deterministic OOM or silently lower the sequence length, model size, or effective batch. Preserve the
failed immutable result. If a capacity investigation is warranted, add a new H100-only `capacity_adjusted` condition
with an explicit objective and do not combine it with E252–E255. If only the four-GPU jobs remain unschedulable,
collect the completed two-GPU evidence and record the four-GPU allocation constraint rather than treating it as a
model or strategy failure.
