# TinyLlama batch versus GPU scaling follow-up

This study asks when extra GPUs improve training throughput, and when the
smaller batch assigned to each GPU leaves the extra capacity underused. It is
a controlled follow-up to E044–E046, with a separate input protocol and
experiment IDs. The earlier runs remain the historical comparison; E044's
observed L40S memory ceiling remains recorded as a failed condition.

## Design

E305–E313 form a 3 × 3 strong-scaling grid. Each row fixes the number of
training examples per optimizer update; each column changes GPU count.

| Global batch | 1 GPU | 2 GPUs | 4 GPUs |
| --- | --- | --- | --- |
| 4 | E305 (4/GPU) | E306 (2/GPU) | E307 (1/GPU) |
| 8 | E308 (8/GPU) | E309 (4/GPU) | E310 (2/GPU) |
| 16 | E311 (16/GPU) | E312 (8/GPU) | E313 (4/GPU) |

The model is pinned TinyLlama 1.1B; dataset is a seeded synthetic global
token stream, sequence length 512, DDP, one node, FP32 model parameters,
BF16 computation, SDPA attention, no KV cache, no activation checkpointing,
AdamW, accumulation 1, learning rate 5e-5, ten warm-up updates, and 200
measured optimizer updates. All ranks consume disjoint slices of the *same*
global examples at a fixed global batch. The input digest must match across
the 1/2/4-GPU runs within a row. New examples are used for each update.
Preparing this stream is outside the timed training interval.

E314–E316 are one-trial acceptance runs at global batch 8 on 1/2/4 GPUs.
E317 checks the largest 1-GPU batch (16) for capacity. They run 50 measured
updates each. The main wave requires all four canaries to have collected,
checksum-valid evidence with correct input identity, optimizer update counts,
batch accounting, dtype, attention backend, rank placement, and measured
energy samples. An out-of-memory canary stops the main wave until the study
plan is adjusted and the failed capacity boundary is recorded. Do not simply
retry the same shape as a throughput experiment.

The diagonal E305/E309/E313 also gives a weak-scaling *view* at constant
4 examples per GPU. It reuses those three trials; do not submit duplicate
jobs or treat the strong and weak views as independent replications. The
analysis groups curves by platform, model, dataset revision, precision,
sequence length, attention backend, workload, runtime version, global input
identity, and the other fixed controls. A missing 1-GPU anchor is reported
as a relative curve, not 1-GPU efficiency.

E318 is an L40S two-GPU serial reproduction of the unusual E291 trial
spread. It retains the original static-input protocol. It records GPU UUIDs,
topology, CPU affinity and visible scheduler jobs for each rank. E319 is a
short profiler diagnostic on the same shape. Its trace overhead makes its
throughput unsuitable for the performance grid. Compare E318 against E291
only after inspecting placement and external contention. Neither one should
be used to infer a causal topology effect from throughput alone.

## On the cluster

Run from the repository root with `source slurm/modules_for_hpc_ai` and
`source .venv/bin/activate`. Commit and pull the complete code and matrix
first; immutable bundles require a clean source tree.

```bash
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-placement-diagnostic
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-placement-diagnostic
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-placement-diagnostic

make submit-scaling-canary PLATFORM=l40s
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-batch-canary
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-batch-canary
make verify-scaling-acceptance PLATFORM=l40s
make submit-scaling-wave PLATFORM=l40s
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-batch-crossover
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-batch-crossover
```

Run the same canary, verification, and main-wave sequence with `PLATFORM=h100`.
H100 does not have the E318/E319 L40S diagnostics. To collect completed
siblings after an incomplete wave, use `PLATFORM_ALLOW_INCOMPLETE=1` and
then a recovery submission once the cause is understood. If canary
verification fails, inspect the canary result JSON and Slurm logs; a
`COMPLETED` scheduler state alone does not establish a valid run.

Within these waves, Slurm permits one array task at a time (`%1`) and starts
each subsequent experiment after the previous array ends. The status CSV
records the dependency chain. This controls contention *within* the wave;
it does not reserve the node against other users or separately submitted
waves. Use a reservation or check node occupancy when interpreting a large
trial spread. Run E319 separately, after the E318 replay, to keep profiling
overhead away from performance trials. Profiler trace files remain in the
submission bundle's `cache/runtime` tree and should be archived separately
if they are needed for the dissertation; the canonical collected JSON
contains the result and trace path.

## Analysis decision

Report throughput, speedup and efficiency with trial counts and 95% intervals,
and show per-GPU memory and measured-region energy alongside them. At fixed
global batch, locate the GPU count where another GPU ceases to improve
throughput enough to justify its cost. Compare those crossover points across
batch sizes and between L40S and H100 only for the same exact workload
contract. The synthetic stream measures execution behavior, not model
generalization or real-data loading. Sequence-length, real-text, and strategy
experiments must retain their own input and environment contracts.
