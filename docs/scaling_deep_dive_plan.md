# Scaling deep-dive study design

**Date:** 2026-10-04 (revised the same day: multi-workload design, memory-safe settings)
**Status:** Implemented in full (S1–S8 plus a run-length check) as E320–E701; see
[scaling_deep_dive_runbook.md](scaling_deep_dive_runbook.md). Departures from this plan: the
TinyLlama FSDP 8 × 1024 arm runs at 4 and 8 GPUs only, because 2 GPUs break the memory rule; S2
adds a second tokens-per-GPU level per workload; and S6 also covers H100 P2P and NCCL algorithms.
**Builds on:** E032–E046, E216–E220, E287–E293 (completed scaling evidence), and the in-progress
TinyLlama batch-crossover wave E305–E319 (configs present, not yet run).

## 1. What the current evidence says

Mean scaling efficiency from `results/analysis/scaling_summary.csv` (exact comparability unless
noted). Efficiency = throughput at N GPUs ÷ (N × throughput at the baseline GPU count).

| Workload (params) | Platform | Weak, max GPUs | Strong, max GPUs |
|---|---|---|---|
| GPT-2 Small (124M) | H100, NVLink | 97% at 4 | 94% at 4 (vs 2) |
| GPT-2 Small (124M) | L40S, PCIe | 93% at 8 | 89% at 8 (vs 4) |
| DistilBERT (66M) | H100 | 92% at 4 | 73% at 4 |
| DistilBERT (66M) | L40S | 78% at 8 | 59% at 8 |
| TinyLlama (1.1B) | H100 | 92% at 4 | 63% at 4 |
| TinyLlama (1.1B) | L40S | 7% at 2, 33% at 4, 40% at 8 | 32% at 8 (vs 2) |

The headline "more GPUs help" holds for small models on either platform. It breaks down for
TinyLlama on L40S, and strong scaling degrades on both platforms as the per-GPU batch shrinks. The
existing runs can't say why, because the workloads also differ in model size, per-GPU batch, sequence
length and task at the same time. For example, TinyLlama ran with batch 2 per GPU on L40S and 8 on H100.

## 2. Working hypothesis

On every optimizer update, data-parallel training does two things: each GPU computes on its own share
of tokens, then all GPUs exchange gradients. The gradient exchange is roughly the same size whatever
the batch, because it scales with the model's parameter count. So:

> **Scaling efficiency is governed by how much compute each GPU does per gradient exchange,
> relative to how long that exchange takes on the link.**

Every parameter worth varying moves one side of that ratio:

- **More compute per exchange:** larger per-GPU batch, longer sequences, gradient accumulation.
- **Fewer bytes per exchange:** a smaller model, gradient compression, or a sharded stack (which also
  adds other traffic).
- **A faster or better-placed link:** NVLink vs PCIe, which GPUs share a PCIe switch, NCCL settings.
- **Noise and imbalance:** stragglers, variable-length inputs, other jobs on the node.

Each study below tests one lever while holding the rest fixed. The interesting results are where a
lever matters much more on one platform or workload than another, or where it doesn't matter at all.

## 3. Workloads: why three, and the memory limits each must respect

TinyLlama alone is a fragile basis for this study: several TinyLlama shapes have already hit CUDA
out-of-memory (OOM), and a TinyLlama-only result can't separate "this is how scaling behaves" from
"this is how a 1.1B model behaves". Every core study therefore runs on three workloads that span a
17× range in parameter count, with GPT-2 Small as the primary workload.

| Workload | Params | Role | Hard limits |
|---|---|---|---|
| **GPT-2 Small** | 124M | **Primary.** Mid-size causal LM. Fits every planned shape on both platforms with margin. | Max sequence 1024 (position embeddings) |
| **DistilBERT** | 66M | Small encoder classifier. Exchanges are tiny, so this end of the range shows latency, not bandwidth, limits. Already shows 59% strong scaling at 8 L40S GPUs. | Max sequence 512. Needs a code change (below) |
| **TinyLlama** | 1.1B | Large end. Where the L40S problem shows up. Kept only at memory-safe shapes. | ~23 GB used before any activations under DDP |

### Measured memory per GPU (from completed runs)

| Workload | Shape (micro-batch × sequence) | H100 | L40S | Notes |
|---|---|---|---|---|
| GPT-2 Small | 8 × 1024 | 15.0 GB | 14.7 GB | |
| GPT-2 Small | 16 × 1024 | 28–30 GB | 28–29 GB | Default shape in past waves |
| GPT-2 Small | 24–25 × 1024 | — | 41–43 GB | L40S ceiling; 26+ OOMed (E203–E207) |
| GPT-2 Small | 32 / 48 × 1024 | 55 / 77 GB | OOM | E067/E068/E257 history |
| DistilBERT | 32 × 512 | 5–6.5 GB | 5–5.7 GB | Large headroom |
| DistilBERT | 128 × 512 | 14.6 GB | 14.3 GB | |
| TinyLlama | 2 × 1024 | — | 25–30 GB | |
| TinyLlama | 4 × 1024 | 27–31 GB | 27–30 GB | Safe on both |
| TinyLlama | 8 × 1024, DDP | 44–46 GB | 42.5 GB | Over the L40S safety margin |
| TinyLlama | 16 × 1024 | 65 GB | OOM | E044 history |

L40S's usable capacity is 44.5 GB and H100's about 79 GB.

### Memory rules for this wave

1. **No planned shape above ~85% of usable capacity** on the platform it runs on: about 38 GB on
   L40S and 67 GB on H100, judged from the table above. When a design needs more work per GPU than
   that, it uses gradient accumulation instead of a bigger micro-batch.
2. **TinyLlama is capped at 4,096 tokens per GPU per micro-step** (for example 4 × 1024), which
   measured 27–31 GB on both platforms.
3. **Every new workload × shape gets a one-trial canary at its largest GPU count on each platform**
   before the three-trial wave, as E314–E317 do for the crossover study.
4. **An OOM is recorded, not dropped.** It goes into `platforms/execution_matrix.json` as a capacity
   deferral, as for E044 and E203–E207, and the dashboard shows it as OOM-limited.

### Code needed for DistilBERT

The controlled scaling protocol (`config/scaling.py` and `GlobalTrainingInputs` in
`workloads/scaling.py`) produces causal-LM batches only (labels = input tokens). DistilBERT needs a
classification variant: one fixed, seeded label per sequence, with the same "preloaded, partitioned,
no duplication" guarantees. This is a small change plus tests. GPT-2 Small can use the protocol as it
is, since it goes through the same causal-LM training loop as TinyLlama.

## 4. Parameters and study rationale

| Parameter | Worth studying? | Why |
|---|---|---|
| `per_gpu_batch_size` | **Yes, core lever** | Directly sets compute per exchange. Partly covered by E305–E313 (TinyLlama only). S2 extends it to all three workloads. |
| `global_batch_size` | Yes, as a consequence | It is GPUs × micro-batch × accumulation, so it moves whenever the others do. Studied through accumulation (S1) and through strong vs weak scaling. |
| `sequence_length` | **Yes** | More tokens per exchange, and attention cost grows faster than linearly, so longer sequences may scale better at the same tokens per GPU (S2). |
| `fixed_workload` (strong vs weak) | Yes | S1 runs both modes on GPT-2 Small; E305–E313 cover the TinyLlama crossover. Other studies use weak scaling, to isolate one lever at a time. |
| `stack` (DDP / FSDP / ZeRO) | **Yes** | Earlier strategy runs compared stacks at fixed GPU counts only. No scaling *curve* exists per stack under controlled inputs (S5). |
| `dataset` | Only in a specific form | For a dense model at fixed sequence length, token content doesn't change the compute, so real vs synthetic text would give the same throughput. What does matter is **variable-length real data**: padding waste and uneven per-GPU work, which fixed-length synthetic inputs hide (S7). |

## 5. Proposed studies

All studies use BF16 compute, three trials per condition, and run on both platforms. GPU counts are
1, 2 and 4 on both platforms, plus 8 on L40S (H100 has 4 GPUs on NVLink; L40S has 8 on PCIe). Each
workload therefore has 3 conditions per row on H100 and 4 on L40S. Proposed IDs start at **E320**.

"Runs today" means the controlled scaling protocol already accepts the configuration (DDP,
synthetic preloaded inputs, BF16 compute with FP32 parameters).

### S1. Gradient accumulation and strong vs weak (GPT-2 and TinyLlama run today; DistilBERT after its protocol change)

- **Question:** If each GPU does several micro-batches before syncing, does efficiency recover where
  it was poor (TinyLlama on L40S, DistilBERT strong scaling)?
- **Design, weak scaling:** accumulation 1, 2, 4, 8 at a fixed micro-batch for each workload:

  | Workload | Micro-batch × sequence | Peak memory expected |
  |---|---|---|
  | GPT-2 Small | 8 × 1024 | ~15 GB |
  | DistilBERT | 32 × 512 | ~6 GB |
  | TinyLlama | 4 × 1024 | ~31 GB |

  Accumulation doesn't increase peak memory, which is why it's the safe way to add work per GPU.
- **Design, strong scaling (GPT-2 Small only):** fixed global batch 64, split across 1–8 GPUs via
  accumulation (1 GPU: 8 × accumulation 8 … 8 GPUs: 8 × accumulation 1). This shows the cost of strong
  scaling without changing micro-batch size or memory.
- **Conditions:** 3 workloads × 4 accumulation values × GPU counts, plus 4 strong-scaling rows.
- **Interesting if:** accumulation alone brings L40S TinyLlama above ~85%, which would show its
  problem is sync frequency, not compute. Also interesting if the gain differs a lot across the three
  model sizes.

### S2. Tokens per GPU: batch vs sequence length (GPT-2 and TinyLlama run today)

- **Question:** Is efficiency governed by tokens per GPU, whatever the batch/sequence split?
- **Design:** fixed tokens per GPU per micro-step, split different ways, weak scaling:

  | Workload | Tokens per GPU | Splits (micro-batch × sequence) |
  |---|---|---|
  | GPT-2 Small | 8,192 | 32 × 256, 16 × 512, 8 × 1024 |
  | DistilBERT | 8,192 | 64 × 128, 32 × 256, 16 × 512 |
  | TinyLlama | 4,096 | 16 × 256, 8 × 512, 4 × 1024 |

  Sequence lengths stay within each model's limits; TinyLlama stays at the 4,096-token cap. GPT-2 at
  32 × 256 should need less memory than 8 × 1024, but its canary must confirm that.
- **Interesting if:** the splits differ by more than ~10 points of efficiency, which would mean
  "tokens per GPU" isn't a good enough rule of thumb. It's also interesting if long sequences help one
  workload and not another.

### S3. Model size: how gradient volume drives scaling (small code change)

- **Question:** At matched work per GPU, how does efficiency fall as parameters grow?
- **Design:** a size ladder within one model family, plus the two existing workloads as cross-family
  checks:

  | Model | Params | Memory risk at 4 × 512 per GPU |
  |---|---|---|
  | DistilBERT | 66M | Low |
  | GPT-2 Small | 124M | Low |
  | GPT-2 Medium | 355M | Low |
  | GPT-2 Large | 774M | Medium: needs a canary on L40S |
  | TinyLlama | 1.1B | Low at this shape |

  2,048 tokens per GPU (4 × 512) is chosen so the largest models fit on L40S with margin; accumulation
  can add work per GPU without adding memory if S2 shows a larger setting is more informative.
- **Code:** register `gpt2-medium` and `gpt2-large` as workloads with pinned revisions.
- **Interesting if:** the efficiency drop with size is sharp on L40S and flat on H100. That would give
  a "model size where PCIe stops being enough" result that practitioners can use directly.

### S4. DDP communication tuning: what helps on PCIe (moderate code change)

- **Question:** Which DDP knobs recover efficiency, and do they matter on NVLink at all?
- **Design:** at S1's accumulation-1 shapes, compare default DDP, larger gradient buckets
  (`bucket_cap_mb=100`), BF16 gradient compression (`bf16_compress_hook`), and both together. GPT-2
  Small and TinyLlama, at 2 and 4 GPUs (+ 8 on L40S).
- **Code:** expose `bucket_cap_mb`, `gradient_as_bucket_view`, `static_graph` and a communication hook
  in the DDP setup (`workloads/transformer.py`), and allow them in the controlled protocol. The
  existing optimizer-update diagnostics check that loss isn't affected. Buckets and compression don't
  raise peak memory meaningfully.
- **Interesting if:** compression helps L40S a lot and H100 not at all, or if it changes the loss or
  update diagnostics.

### S5. Stack scaling curves: DDP vs FSDP (larger code change)

- **Question:** Does sharding scale better or worse than DDP as GPUs are added, under the same inputs?
- **Design:** DDP, FSDP `shard_grad_op` and FSDP `full_shard` on GPT-2 Small and TinyLlama at a fixed
  micro-batch. Optional second arm: each stack at the largest micro-batch it fits within the memory
  rule. This arm is where TinyLlama benefits: FSDP cut its memory from 44 GB to 31–35 GB in E278–E284,
  so it can run shapes DDP can't.
- **Code:** extend the controlled protocol beyond DDP; FSDP support already exists in the runner.
- **Interesting if:** the curves cross. For example, FSDP scales worse at a fixed batch, but wins once
  it's allowed the larger batch it makes possible.

### S6. L40S placement and topology (diagnostic; waits on E318/E319)

- **Question:** Does it matter which L40S GPUs a job lands on? E291's 2-GPU result (7% efficiency)
  is the motivating anomaly.
- **Design:** 2-GPU pairs on the same PCIe switch vs across CPU sockets; NCCL P2P on vs off. GPT-2
  Small first, because it's cheap and memory-safe. Requires a full-node allocation and explicit
  `CUDA_VISIBLE_DEVICES`.
- **Gate:** only design this after E318/E319 show whether E291 was placement, node contention, or
  something else.

### S7. Variable-length real data: padding and stragglers (larger code change, lower priority)

- **Question:** Do realistic variable-length inputs cost scaling efficiency that synthetic data hides?
- **Design:** a real text dataset on GPT-2 Small, comparing pad-to-max, sequence packing and
  unbucketed variable lengths. Report step-time p95 as well as means, to show straggler effects.
- **Code:** a new input protocol alongside `synthetic_global_batches_v1`.
- **Interesting if:** unbucketed variable lengths lose efficiency that packing recovers. That's a
  practical "do this, not that" finding.

### Optional S8. Faster compute scales worse?

FP32 vs BF16 (and FP8 on H100) at fixed batch. Faster per-GPU compute leaves the same exchange
exposed, so lower precision may show *lower* scaling efficiency despite higher absolute throughput.
That's a counter-intuitive and quotable result, but it needs the controlled protocol's precision
contract relaxed, and FP32 raises memory (GPT-2 Small FP32 measured 34 GB vs 28 GB at 16 × 1024).

## 6. Measurement additions (apply to every study)

- **Exposed communication per step:** compare step time at N GPUs against the matched 1-GPU step
  time. The runner already records `step_time_mean_ms`, `step_time_p50_ms` and `step_time_p95_ms`.
- **Straggler signal:** report p95 ÷ p50 step time; flag conditions above an agreed threshold rather
  than averaging them away (the approach taken with E291).
- **Peak memory per condition**, checked against the memory rule, so the next wave's shapes come from
  measurement rather than estimates.
- **Run length check:** one long run (for example 500 updates) per platform at the worst S1 setting,
  to confirm 100 measured updates is enough.
- **GPU-hours per useful token:** alongside efficiency, because what a practitioner pays for is cost to
  train, not efficiency itself.

## 7. Sequencing and gates

| Step | Contents | Gate before the next step |
|---|---|---|
| 0 | E314–E317 canaries, E305–E313 crossover, E318/E319 diagnostics (already configured) | Canaries pass; E291 cause understood well enough to rule placement in or out |
| A0 | DistilBERT classification support in the controlled protocol; one-trial canaries for every new workload × shape | All canaries complete within the memory rule; any OOM recorded and its shape replaced |
| A | **S1 and S2** on all three workloads | Pick the tokens-per-GPU setting for S3 from the S2 results |
| B | **S3 and S4** (small/moderate code) | Loss parity confirmed for compression; GPT-2 Medium/Large weights cached on both clusters |
| C | **S5**, then S6/S7 as the results justify | — |

**Rough cost**, assuming about 0.15 GPU-hours per GPU per trial (measured loops run 20–90 seconds;
model loading and job start-up dominate). The second and third workloads roughly triple Wave A
compared with the TinyLlama-only plan:

| Wave | Contents | H100 GPU-h | L40S GPU-h |
|---|---|---|---|
| A0 canaries | ~20 one-trial runs at max GPU count | ~5 | ~15 |
| A (S1 + S2) | 3 workloads | ~70 | ~150 |
| B (S3 + S4) | 5-model ladder + tuning | ~35 | ~75 |
| C (S5) | 2 workloads × 3 stacks | ~20 | ~45 |

If that's too much, the cheapest cut is running S2 on GPT-2 Small and TinyLlama only, since
DistilBERT is the workload that needs new code.

## 8. Decisions needed before writing configs

1. **Matched or platform-native batch sizes?** The plan matches micro-batches across platforms, which
   makes H100 vs L40S comparisons clean. Using each platform's largest feasible batch is more
   realistic, but repeats the confound in the current TinyLlama results.
2. **DistilBERT in or out of Wave A?** It needs the classification change first. Leaving it to Wave B
   lets Wave A start with configs only.
3. **Start Wave A now, or after E305–E319 complete?** S1 and S2 don't depend on the crossover results,
   but the crossover may change which TinyLlama shape is most informative.
4. **Which of S3–S7 are in scope** for the time left before writing up?
