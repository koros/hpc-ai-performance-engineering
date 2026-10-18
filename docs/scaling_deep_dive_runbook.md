# Scaling deep dive: runbook

**Study ID:** `scaling_deep_dive_v1` · **Conditions:** E320–E701 (382) · **Design:** [scaling_deep_dive_plan.md](scaling_deep_dive_plan.md)

This study takes the headline finding "more GPUs help, except TinyLlama on L40S" apart one lever at a
time, on three main workloads (GPT-2 Small, DistilBERT, TinyLlama) plus GPT-2 Medium and Large for
the model-size ladder, on both platforms. Every condition uses the controlled scaling protocol:
identical, pre-loaded inputs for every GPU count; optimizer-update diagnostics; recorded GPU
placement; and serial submission so a wave never competes with itself on a node.

The in-progress TinyLlama batch-crossover wave (E305–E319) is a separate study and is not part of
this one.

## 1. What each wave tests

| Study | Wave | What it varies | IDs | H100 conditions | L40S conditions | H100 GPU-h | L40S GPU-h |
|---|---|---|---|---|---|---|---|
| C | `scaling-dd-canary` | One short trial per distinct memory/backend risk, at the least-sharded GPU count | E320–E385 | 63 | 66 | ~11 | ~13 |
| S1 | `scaling-dd-accumulation` | Gradient accumulation 1/2/4/8 × GPUs 1/2/4/8; its anti-diagonals are strong-scaling curves at a fixed global batch | E386–E433 | 36 | 48 | ~52 | ~112 |
| S2 | `scaling-dd-split` | Batch × sequence split at fixed tokens per GPU (two token levels per workload), plus a GPT-2 batch axis at sequence 1024 | E434–E505 | 54 | 72 | ~44 | ~94 |
| S3 | `scaling-dd-model-size` | DistilBERT, GPT-2 Small/Medium/Large and TinyLlama at 4 × 512, accumulation 1 and 4 | E506–E533 | 21 | 28 | ~28 | ~59 |
| S4 | `scaling-dd-ddp-comm` | DDP buckets 5/100 MB, BF16 gradient compression, both together, and bucket-view + static graph | E534–E578 | 30 | 45 | ~31 | ~73 |
| S5 | `scaling-dd-stacks` | FSDP `sharded_grad_op` and `full_shard` (per-layer wrap), `full_shard` root-only wrap, and TinyLlama 8 × 1024 that only FSDP fits on L40S | E579–E607 | 19 | 29 | ~20 | ~48 |
| S6 | `scaling-dd-topology` | L40S: pinned GPU sets (same PCIe switch / same socket / across sockets), P2P off, NCCL Ring vs Tree. H100: P2P off, Ring vs Tree | E608–E637 | 8 | 22 | ~10 | ~62 |
| S7 | `scaling-dd-real-text` | WikiText-103 padded to full length, padded per micro-batch, length-bucketed, or packed | E638–E669 | 24 | 32 | ~20 | ~42 |
| S8 | `scaling-dd-precision` | FP32 vs BF16 autocast; Transformer Engine BF16 vs FP8 | E670–E697 | 21 | 28 | ~17 | ~37 |
| RL | `scaling-dd-run-length` | 1,000 vs 200 measured updates at the worst-scaling shapes | E698–E701 | 2 | 4 | ~6 | ~18 |
| | | **Total** | | **278** | **374** | **~240** | **~560** |

GPU-hour figures are planning upper bounds: about five minutes of start-up plus the measured loop,
per reserved GPU, three trials per condition (one per canary). The measured loops themselves run
for one to three minutes.

28 conditions serve more than one study. For example, the GPT-2 8 × 1024 accumulation-1 runs are
also an S2 split point and the S8 BF16 control. Each is generated and run once, in the wave of
the first study that needs it.

## 2. Design rules the generator enforces

- **Memory:** no condition's estimated peak exceeds 85% of usable capacity on any platform it runs
  on (about 38 GB on L40S, 67 GB on H100). The estimate is conservative: about 35 GB for TinyLlama
  4 × 1024, which measured 30–31 GB. More work per GPU comes from accumulation, which doesn't raise
  peak memory. TinyLlama stays at 4,096 tokens per GPU per micro-step except under FSDP, and its
  FSDP 8 × 1024 arm runs at 4 and 8 GPUs only.
- **Platforms:** 8-GPU conditions and pinned placement run on L40S only; H100 has 4 GPUs.
- **Model limits:** sequence length ≤ 1024 for GPT-2 and ≤ 512 for DistilBERT.
- **Inputs:** every GPU count consumes the same ordered global examples. Synthetic runs use
  `synthetic_global_batches_v1` (DistilBERT gets one seeded label per sequence). S7 uses
  `wikitext103_global_batches_v1` at the WikiText revision already pinned for the low-bit study.
- **Static graph:** `ddp_static_graph` is rejected with gradient accumulation. PyTorch's reducer
  asserts when `no_sync` skips a static graph's synchronisation; a CPU smoke run reproduced this.
- **FP8 controls:** every FP8 condition names its matched Transformer Engine BF16 condition in
  `comparison_control_id`.

`make check-scaling-deep-dive` (and the test suite) fails if a config or wave drifts from the design.
To change the design, edit `src/hpc_ai_perf/config/scaling_deep_dive.py`, then run
`make generate-scaling-deep-dive`. Don't hand-edit `configs/scaling_deep_dive/` or the `scaling-dd-*`
waves.

## 3. How to run it

1. **Cache assets on the login node** (models, tokenizers, WikiText-103), so no job downloads
   during its reservation:

   ```bash
   make prefetch-scaling-deep-dive-assets
   ```

2. **Canaries first, on both platforms.** Each canary is a single 20-update trial:

   ```bash
   make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-dd-canary
   ```

   ```bash
   make submit-platform-wave PLATFORM=h100 PLATFORM_WAVE=scaling-dd-canary
   ```

   Then use `status-platform-wave` and `collect-platform-wave` with the same arguments.
3. **Gate before the three-trial waves.** Collect the canary wave, then run
   `make verify-scaling-deep-dive-canaries PLATFORM=l40s` (or `PLATFORM=h100`).
   Performance-wave submission also enforces this collected-evidence gate and fails before
   creating a submission bundle if a canary is absent or invalid. For each platform, every canary must be completed, with
   `memory_used_gb` below 85% of capacity, `optimizer_step_calls == expected_optimizer_step_calls`,
   and `attention_backend_observed == sdpa`. Pay particular attention to:
   - DistilBERT canaries: the first runs of the classification inputs and SDPA attention.
   - Transformer Engine BF16/FP8 canaries: the first multi-GPU Transformer Engine runs anywhere in
     the project.
   - BF16-compression canaries: these need NCCL ≥ 2.10.
   - FSDP canaries: the first per-layer wrapping of GPT-2, DistilBERT and TinyLlama.
   - The pinned-placement canaries (L40S only): check `training_rank_placement` reports the
     requested GPUs.

   **L40S placement recovery (E368/E369):** These two-rank conditions reserve all eight GPUs,
   while `nvidia-smi` may report all eight even when `CUDA_VISIBLE_DEVICES` exposes only two to
   PyTorch. The profiler now queries the CUDA driver for the two visible UUIDs when PyTorch does
   not expose them, then selects matching `nvidia-smi` rows. It still fails closed if the driver
   and PyTorch visible-device counts differ or the UUIDs cannot be matched; it never assumes the
   first two `nvidia-smi` rows are the requested GPUs. Check the mapping on an eight-GPU allocation
   before retrying the failed canaries:

   ```bash
   srun -p gpu_l40s --gres=gpu:8 --cpus-per-task=8 --time=00:05:00 bash -lc '
   source slurm/modules_for_hpc_ai
   source .venv/bin/activate
   export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
   python - <<PY
   from hpc_ai_perf.profiling.system import (
       NvidiaSmiProfiler, query_nvidia_smi_rows, select_nvidia_smi_rows,
       visible_cuda_device_uuids,
   )
   uuids = visible_cuda_device_uuids(2)
   rows = query_nvidia_smi_rows(NvidiaSmiProfiler._QUERY)
   selected, method = select_nvidia_smi_rows(rows, target_gpu_uuids=uuids, expected_gpu_count=2)
   print("CUDA UUIDs:", uuids, "nvidia-smi rows:", len(rows), "mapping:", method)
   assert len(uuids) == len(selected) == 2 and method == "cuda_uuid"
   PY
   '
   ```

   Use `make retry-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-dd-canary` only after this
   probe passes, with the same runtime activation used in the original submission. Collect the
   recovery attempt and rerun the canary gate; preserve the original failed bundle as diagnostic
   evidence.

   If a canary OOMs or fails, preserve its failed evidence and adjust the design and regenerate.
   Deferring a canary alone does not open the performance waves: the gate requires the full
   declared canary set, so the matching performance condition must also be removed or deferred
   through a reviewed design change. Never bypass the gate to submit the original risky shape.
4. **Run the studies in order**, collecting each wave before the next. S1 and S2 come first,
   because S3's reading depends on S2's split results:

   ```bash
   make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-dd-accumulation
   ```

   Repeat for `split`, `model-size`, `ddp-comm`, `stacks`, `topology`, `real-text`, `precision` and
   `run-length`, on each platform (H100 has no conditions that need 8 GPUs or pinned placement).
   Use `retry-platform-wave` for incomplete trials.
5. **Refresh and read the analysis:**

   ```bash
   make refresh
   ```

## 4. What the analysis produces

`make refresh` writes three outputs to `results/analysis/` once deep-dive trials exist:

- **`scaling_deep_dive_conditions.csv`** has one row per condition per platform: throughput with a
  95% CI and CV; scaling efficiency against the matched 1-GPU run, with a conservative CI;
  **exposed communication** (step time at N GPUs minus the matched 1-GPU step time) and its share of
  the step; **straggler ratio** (p95 ÷ p50 step time); **GPU-hours per million tokens**; peak memory
  and its share of capacity; padding fraction and per-rank token imbalance (S7); OOM and failed
  trials; and the **placement class** of each trial (`same_pcie_switch`, `same_socket`,
  `cross_socket`, `nvlink`), read from the topology the trial captured itself.
- **`scaling_deep_dive_effects.csv`** has one row per lever contrast: treatment vs control at the
  same GPU count with everything else equal, the throughput change with a Welch 95% CI, the change
  in efficiency points, and a verdict. A change is `helps`/`hurts` only if it is ≥ 3% and its CI
  excludes zero, and `no_material_effect` if its CI sits inside ±3%. Anything else is
  `inconclusive`. Levers are inferred from the parameters that actually ran: accumulation, batch ×
  sequence split, DDP settings, training stack, precision, FP8, text batching, real vs synthetic
  text, and NCCL/placement.
- **`scaling_deep_dive_report.md`**: generated findings covering coverage, a verdict tally per lever
  and platform, the 20 largest material effects, efficiency at the largest GPU count, throughput by
  L40S placement class, and straggler flags.

`scaling_summary.csv` also picks up every deep-dive run as both a strong curve (fixed global batch;
accumulation may vary along it) and a weak curve (fixed micro-batch and accumulation).

## 5. Known risks and how they're handled

| Risk | Handling |
|---|---|
| Multi-GPU Transformer Engine (S8) has never run in this project | Canary-gated; the scaling contract restricts it to DDP on GPT-2/DistilBERT |
| DistilBERT SDPA support depends on the cluster's Transformers version | Canary records `attention_backend_observed`; a failure is visible before the waves |
| Pinned placement assumes the L40S node layout captured by E215: switch pairs (0,1), (2,3), (4,5), (6,7); GPUs 4–7 on socket 1 | `CUDA_DEVICE_ORDER=PCI_BUS_ID`; each trial records its own topology and GPU IDs, and the analysis classifies from that record, not from the assumption |
| Unpinned L40S runs land wherever Slurm puts them | Placement class is recorded for every trial and summarised in the report, so a bad landing shows up instead of being averaged away |
| Serial waves take long wall-clock time | Deliberate: no self-contention. Waves can still run on H100 and L40S at the same time |
| Real-text throughput isn't comparable with synthetic nominal tokens | Real-text runs report *useful* (non-padding) tokens as `throughput_tokens_sec`, plus processed and nominal tokens separately |

## 6. Verification done before submission

- All 382 configs pass the capability checks and the scaling contract, and render Slurm scripts on
  both platforms (278 H100 and 374 L40S jobs). Placement jobs reserve `gpu:8` and 64 CPUs but start
  2 or 4 ranks on the pinned GPUs.
- The full test suite passes (1,101 tests, including the deep-dive tests).
- CPU smoke runs through the real runner using tiny Hugging Face models:
  - GPT-2 and DistilBERT controlled inputs: correct accumulation accounting, SDPA observed.
  - 2-process DDP over gloo with custom buckets, bucket view and static graph.
  - All four WikiText-103 batching modes with a real tokenizer.
  - The analysis computed over those real result records.

  What a CPU smoke run can't show: BF16 compression (needs NCCL), FSDP and Transformer Engine
  (need CUDA), and true memory figures. The canary wave covers those.

## Fresh canaries for reproducible campaigns (16 October 2026)

The canary gate requires an exact match to the current four-file runtime
fingerprint. There are no historical-runtime exceptions. Inference changes in the
shared `workloads/transformer.py` file can change this fingerprint even when the
training definitions are unchanged. Preserve previous results and their original
SHA256SUMS as historical evidence; never relabel them with the current fingerprint.

For the new campaign, commit and freeze the benchmark source before submission.
Use a clean HPC checkout and the dedicated Transformer Engine activation wrapper
for the full canary wave, including its BF16 controls. Do not change fingerprinted
runtime files between canaries and the performance waves they authorize.

```bash
git pull --ff-only
git status --short  # Review any changes before creating the frozen bundle.
FP8_ACTIVATE="$HOME/.local/share/hpc-ai-performance-engineering/fp8-training-te2.8.0/bin/activate_hpc_ai_fp8"
test -r "$FP8_ACTIVATE"
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-dd-canary \
  VENV_ACTIVATE="$FP8_ACTIVATE"
# After all jobs finish:
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-dd-canary
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=scaling-dd-canary
make verify-scaling-deep-dive-canaries PLATFORM=l40s
```

Use `submit-platform-wave`, without experiment filters, to deliberately rerun all
66 L40S canaries. `retry-platform-wave` can reuse historical completed evidence
and is unsuitable for requesting this fresh campaign. Collection preserves earlier
attempts and imports the new completed evidence. Submission of performance waves
remains gated until the collected current-runtime canaries pass all checks.
H100 needs its own full set of 63 collected canaries; L40S does not authorize H100.
Use the dedicated runtime for subsequent Transformer Engine conditions too.

Run unfinished performance waves one at a time after acceptance. Existing trials
remain historical evidence; obtain fresh matched trials when comparisons require
a consistent implementation/environment. Publish source commit, runtime fingerprint,
resolved configs, environment manifests, original JSON and checksums with the results.

## Queue a wave behind another wave

`submit-platform-wave --after-job JOB_ID` adds a Slurm `afterany` dependency.
For a serial study wave, its first job waits for the supplied job/array and all
subsequent jobs retain their normal serial chain. Use the final submitted parent
array ID from the preceding serial wave's `SLURM_STATUS.csv`, not an individual
array task. Confirm that all preceding scripts were submitted before chaining.
The dependency is recorded in the new submission CSV. Do not submit the same
follow-on wave twice.

For example, queue `scaling-dd-ddp-comm` behind the final
`scaling-dd-model-size` array using the CLI's `--after-job` option and the usual
`--platform-profile platforms/l40s.json --wave scaling-dd-ddp-comm
--venv-activate /absolute/path/to/bin/activate_hpc_ai_fp8` arguments.
This schedules S4 after S3 terminates regardless of success; it prevents overlap
but does not establish that S3 benchmark JSON passed validation. Review and
collect both waves after completion. The full collected canary gate still runs
before performance submission. Changing this scheduler option does not change
the training runtime fingerprint. Slurm must still know the predecessor job ID
when the new wave is submitted.
