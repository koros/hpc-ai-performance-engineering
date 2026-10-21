# TinyLlama microbatch versus accumulation follow-up

This is a **follow-up motivated by observed results**, planned on 2026-10-21
following the TinyLlama accumulation results E418–E433, available by results
commit `11bdd72`. It is exploratory sequential research, not an original
preregistered hypothesis. No historical evidence is modified.

## Question and controlled design

At the same effective batch per GPU and the same sequence length, does a larger
microbatch improve throughput, energy efficiency or weak scaling compared with
more gradient accumulation?

| Microbatch per GPU | Accumulation | Effective batch per GPU | Estimated peak GiB |
|---:|---:|---:|---:|
| 1 | 16 | 16 | 24.2 |
| 2 | 8 | 16 | 27.9 |
| 4 | 4 | 16 | 35.3 |

Estimates use the existing conservative DDP memory model, not measured peaks.
The proposed 8/2 point is excluded: approximately 50.1 GiB exceeds the existing
85% of 44.52 GiB usable-memory rule. Do not change precision, checkpointing or
sharding just to fit that point: it would introduce another treatment.

Pinned TinyLlama-1.1B model revision, sequence length 1,024, DDP/NCCL, FP32
parameters with BF16 autocast, SDPA, default gradient communication, seed 1337,
learning rate 5e-5, no activation checkpointing, and synthetic global input
protocol remain constant. All pairs have 16 sequences / 16,384 input tokens per
GPU per optimizer update. Global batch is `16 * GPUs`: these are **weak-scaling**
curves, not fixed-global-batch strong scaling. Input token counts include the
protocol's existing causal-LM counting convention, not a new loss-token metric.

Within a GPU count, changing the split consumes the same seeded global examples
for the same number of optimizer updates. The runtime already averages loss
over accumulation steps and uses DDP `no_sync` on intermediate microsteps.
Microbatching may still change floating-point ordering and stochastic layers;
this measures system performance, not numerical equivalence or time to quality.

## Waves and GPU-time controls

- `microbatch-accumulation-canary`: E773–E778, all three pairs at 1 and 8 GPUs;
  10 warmup + 20 measured updates, one trial each (6 trials).
- `microbatch-accumulation-performance`: E779–E790, all three pairs at
  1, 2, 4 and 8 GPUs; 10 warmup + 200 measured updates, three trials each
  (36 trials). Submission is serial through the existing platform-wave machinery.

The 4/4 conditions repeat the shape of historical E426–E429 intentionally.
Their historical runtime fingerprints differ from the current implementation;
old measurements are context, not matched denominators. Fresh controls share
this campaign's runtime. No other historical wave needs rerunning. All runtime
files used by the existing fingerprint remain unchanged by this addition.

The six acceptance runs must pass checksums, resolved configuration identity,
current runtime fingerprint, clean source, GPU/world size, finite loss,
optimizer/backward accounting, throughput and measured memory ceiling. They
must have matching library versions and identical global input digests across
microbatch splits at the same GPU count. The latest matching attempt controls
acceptance, so an older success cannot conceal a failed retry. Submission of
the performance wave enforces this gate before creating a bundle.

The existing planning estimates are conservative, not a promised wall time.
Review measured canary timings before committing to the full wave. Synthetic
inputs are prepared before timing; end-to-end data-loader performance is outside
this study. Trials use the existing repeated-seed protocol; confidence intervals
quantify repeat variation, not variation across training seeds.

## Run on the HPC

Commit/push the implementation locally, then pull it on the HPC. From its repo root:

```bash
source slurm/modules_for_hpc_ai
source "$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo-recovered/bin/activate_recovered_vllm"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
RECOVERED_ACTIVATE="$VIRTUAL_ENV/bin/activate_recovered_vllm"

python scripts/generate_microbatch_accumulation.py --check
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json \
  --wave microbatch-accumulation-canary \
  --venv-activate "$RECOVERED_ACTIVATE"
```

This uses the recovered PyTorch environment; vLLM is not the training backend and
Transformer Engine is not required. These TinyLlama training acceptance runs
validate the recovered environment for this workload rather than assuming the
inference smoke test certified training.

Track, then collect only after all 6 trials finish:

```bash
python -m hpc_ai_perf.cli status-platform-wave \
  --platform-profile platforms/l40s.json --wave microbatch-accumulation-canary
python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave microbatch-accumulation-canary
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
python scripts/verify_microbatch_accumulation_canaries.py
```

If acceptance passes and its timings are acceptable, submit performance:

```bash
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json \
  --wave microbatch-accumulation-performance \
  --venv-activate "$RECOVERED_ACTIVATE"
```

Use that wave name in the same status/collect commands. After collecting all
36 performance trials, verify evidence before staging:

```bash
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
python -m hpc_ai_perf.cli stage-platform-wave \
  --platform-profile platforms/l40s.json --wave microbatch-accumulation-performance
# Review the staged changes before committing.
git diff --cached --stat
```

Do not pass `--result-root` to `stage-platform-wave`: its parser does not accept it.

## Analysis to perform after collection

Report throughput, peak memory, energy per processed token and update latency
alongside efficiency. For each split use its own matched one-GPU baseline:
`efficiency(G) = throughput(G) / (G * throughput(1))`. Keep runtime/configuration
cohorts separate and require complete, valid repeats for conclusions.

Compare splits within each GPU count: effective global batch, sequence length,
input digest and update count must match. A faster microbatch at equal effective
batch is a different finding from a better efficiency ratio. Higher efficiency
alone does not establish higher absolute throughput or faster convergence.

Generic CSV/JSON condition summaries from `make export-plot-data PYTHON=python`
will include these performance results. Filter by
`study == "tinyllama_microbatch_accumulation_v1"`; the historical deep-dive-only
scaling tables intentionally retain their original scope. Plot throughput and
energy against microbatch, memory against microbatch, and efficiency against GPU
count with a separate curve for each split. Use three trial repetitions, not
optimizer updates or tokens, as the sampling unit.
