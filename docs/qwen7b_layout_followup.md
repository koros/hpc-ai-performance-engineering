# L40S Qwen7B equal-budget layout follow-up

The completed TinyLlama campaign has 48 performance trials. Replicas won
throughput at both tested prompt lengths and concurrency endpoints; at concurrency
8, TP2 replicas used less measured-region energy per output token than TP1.
The new study tests whether that throughput–energy trade-off persists with a
larger model. It does not establish the ranking of TP4/TP8 or an exact concurrency
crossover. Do not rerun historical TinyLlama conditions or pool the two models.

Pinned model: Qwen/Qwen2.5-7B-Instruct at
`a09a35458c702b33eeacc393d103063234e8bc28` (existing E301 model revision).
All conditions use eight L40S GPUs, BF16, vLLM V0/eager, no prefix caching,
2,048 context length, 0.85 GPU memory utilisation, synthetic prompts, concurrency
8, exactly 128 output tokens per request, seed 1337, and request timestamps.
Performance uses 30 measured rounds after five warmup rounds; acceptance uses
2+2 rounds. Conditions are serial. Runtime paths are recorded in bundle contracts;
use the recovered environment already tested on L40S.

| Wave | IDs | Layout/prompt | Trials |
|---|---|---|---:|
| qwen7b-layout-canary | E750 | TP1 × DP8, prompt 1280 | 1 |
| qwen7b-layout-canary | E751 | TP2 × DP4, prompt 1280 | 1 |
| qwen7b-layout-performance | E752/E753 | TP1 × DP8 / TP2 × DP4, prompt 128 | 6 |
| qwen7b-layout-performance | E754/E755 | TP1 × DP8 / TP2 × DP4, prompt 1280 | 6 |

Total: two acceptance and twelve performance trials. GPU-hour values in configs
are scheduling allowances, not measured duration predictions. Use first-canary
startup/memory/time evidence before committing the rest of the allocation.

## Preparation and first trial

Push/pull the committed study. On HPC, source the modules and recovered activation.
Keep the source and original private package tree unchanged throughout this study.

```bash
source slurm/modules_for_hpc_ai
source "$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo-recovered/bin/activate_recovered_vllm"
export PYTHONPATH="$PWD/src"
python scripts/generate_qwen_layout.py --check
python scripts/prepare_qwen_layout_model.py
# If cache verification fails, prefetch on the LOGIN node, then verify again:
# python scripts/prepare_qwen_layout_model.py --prefetch
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-canary \
  --experiment-id E750 \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

The runtime configs force offline model loading; model download must finish on the
login node before GPU submission. The first trial exercises eight independent
engines: GPU fit does not establish adequate host RAM/CPU/startup capacity. Inspect
completed status, errors, timestamps and routing before submitting only E751.
Collect the first trial with `--allow-incomplete` before updating the wave state.

After both canaries finish, collect again with `--allow-incomplete`, run
`scripts/verify_qwen_layout_canaries.py`, verify result evidence, and stage/commit
acceptance. Latest-bundle status may list the earlier canary as missing; the gate
checks the combined collected records. Never rerun it solely for that display.
The model-specific gate rejects wrong model/config/checksum/runtime, missing
TTFT/decode metrics, incorrect work/GPU counts and unbalanced acceptance routing.
TinyLlama canaries cannot authorize this performance wave.

## Performance

Only after acceptance passes, submit `qwen7b-layout-performance` with the same
platform profile and recovered activation; no experiment filter. The CLI enforces
the gate before creating a submission bundle. Collect the twelve trials, validate
and commit using the ordinary platform-wave lifecycle. Use the per-prompt TP1
controls E752/E754 for TP2 contrasts; report means/uncertainty across independent
trials, within-trial latency percentiles, measured-region J/output token, and
reserved GPU time separately. No direct PCIe/NVLink attribution or quality claims.

Local regeneration: `python scripts/generate_qwen_layout.py`; validation:
`make check-qwen7b-layout`. The existing TinyLlama generator and serving fingerprint
remain unchanged. Recovery diagnostics are not benchmark acceptance evidence.

## TP4 extension after the completed TP1/TP2 campaign

The twelve performance trials showed TP2 winning throughput/energy for Qwen7B,
with a long-prompt TTFT trade-off. Add TP4 × DP2 at the same eight-GPU budget,
concurrency 8, two prompt lengths and fixed output budget. Existing E750–E755
configs, comparison scope and serving runtime remain unchanged; do not rerun them.

**TP8 is unsupported for this model:** the [pinned model configuration](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json)
has 28 query heads, which divide evenly by TP1/TP2/TP4 but not TP8.
The extension covers all these valid equal-budget power-of-two TP layouts;
it does not claim exhaustive coverage of arbitrary GPU counts or concurrency.

| Wave | ID | Shape | Trials |
|---|---|---|---:|
| qwen7b-layout-extension-canary | E756 | TP4 × DP2, prompt 1280 | 1 |
| qwen7b-layout-extension-performance | E757 | TP4 × DP2, prompt 128 | 3 |
| qwen7b-layout-extension-performance | E758 | TP4 × DP2, prompt 1280 | 3 |

Seven new trials total. Performance uses existing TP1 controls E752/E754 and
can also be compared with matched TP2 E753/E755. Repeat the login-node cache
verification from above; no new model download is needed if its cache is intact.
Check HPC state before submitting to avoid duplicating an existing attempt.

With the modules and recovered environment active, run from the project root:

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_qwen_layout_model.py
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-extension-canary \
  --experiment-id E756 --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

When E756 is complete:

```bash
python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-extension-canary
python scripts/verify_qwen_layout_canaries.py --extension
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
```

Stage and commit this acceptance before performance submission. The extension
performance gate requires both original Qwen acceptance and new TP4 acceptance;
it does not require fresh TP1/TP2 trials. Then submit:

```bash
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-extension-performance \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

Collect/validate/stage the six performance trials before analysis. Compare the
three layouts within each matched prompt shape; preserve each independent trial.
The extension is sequential in time, so inspect telemetry/clocks/runtime identity
against the earlier controls before attributing differences to layout alone.

## Concurrency-one gap study

With all three layouts measured at concurrency 8, test whether the preference
changes at concurrency 1. Use only 1,280-token prompts and 128 fixed output tokens,
eight GPUs, the same pinned Qwen7B revision, BF16/V0/eager and cache settings.
No completed inputs or serving-runtime fingerprint are changed.

| Wave | IDs | Layouts | Trials |
|---|---|---|---:|
| qwen7b-layout-low-concurrency-canary | E759/E760/E761 | TP1×DP8 / TP2×DP4 / TP4×DP2 | 3 |
| qwen7b-layout-low-concurrency-performance | E762/E763/E764 | Same three layouts | 9 |

Acceptance uses two warmup and eight measured singleton rounds: eight requests
exercise every replica and permit strict balanced-routing acceptance. Performance
uses five warmup and thirty measured rounds, three repetitions per condition.
Thirty singleton requests cannot divide equally over eight/four replicas: a
one-request routing difference is expected, not a failure. Each trial emits
3,840 output tokens. Trial-level p99 from thirty requests is exploratory.

Within concurrency 1, E762 is the TP1 control. Existing E754/E755/E758 provide the
matched concurrency-8 references; the new configs record these in metadata.
The prompt/output/model/GPU budget remain fixed. Compare layout rankings and
throughput, first-token latency, decode pace and measured-region J/output token.
Do not describe lower whole-node utilisation at concurrency 1 as a software
failure: idle replicas are part of the fixed-resource-budget comparison.

Submit only the three serial acceptance jobs first, with the recovered activation:

```bash
export PYTHONPATH="$PWD/src"
python scripts/prepare_qwen_layout_model.py
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-low-concurrency-canary \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

After all three finish:

```bash
python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-low-concurrency-canary
python scripts/verify_qwen_layout_canaries.py --low-concurrency
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
```

Stage/commit acceptance. The low-concurrency performance wave requires original,
TP4-extension and all three singleton canaries (six accepted records total).
Then submit `qwen7b-layout-low-concurrency-performance` using the same recovered
activation; expect nine trials. Collect/validate/commit before deciding whether
intermediate concurrency 2/4 would add useful evidence. Do not queue those
follow-ups automatically. The sequential-in-time caveat for prior controls still
applies; compare runtime identities, clock and telemetry before strong attribution.

## Concurrency-four energy ranking follow-up

The concurrency-one performance trials slightly favoured TP2 throughput and
used less J/output token than TP4. At concurrency 8, TP4 used less J/output token.
Measure the intermediate concurrency **4**, retaining prompt 1,280, output 128,
eight L40S GPUs, identical model revision, precision, seed, runtime and telemetry.
This narrows the observed energy-ranking reversal; it does not measure an exact
threshold or assume a monotonic relationship. Concurrency 2 remains unmeasured.

| Wave | IDs | Layouts | Trials |
|---|---|---|---:|
| qwen7b-layout-mid-concurrency-canary | E765/E766 | TP2×DP4 / TP4×DP2 | 2 |
| qwen7b-layout-mid-concurrency-performance | E767/E768 | Same | 6 |

Canaries use two warmup and two measured rounds (eight requests total); both
layouts receive perfectly balanced routing. Performance uses five warmup and
thirty measured rounds, three repetitions: 120 requests / 15,360 output tokens
per trial. E767 is the within-concurrency TP2 control for E768. Metadata records
the matched concurrency-one and concurrency-eight reference IDs.
Completed configurations and the serving runtime fingerprint remain unchanged;
do not rerun prior trials. Keep the recovered runtime and private package tree
fixed, and inspect telemetry against prior runs when interpreting comparisons.

With modules/recovered activation loaded and `PYTHONPATH="$PWD/src"`:

```bash
python scripts/prepare_qwen_layout_model.py
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-mid-concurrency-canary \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

After both finish, collect, then verify:

```bash
python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave qwen7b-layout-mid-concurrency-canary
python scripts/verify_qwen_layout_canaries.py --mid-concurrency
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
```

The verifier requires eight accepted Qwen canaries across all completed shapes.
Stage/commit acceptance, then submit `qwen7b-layout-mid-concurrency-performance`
with the same activation. Expect six performance trials. The CLI enforces the
shape-specific acceptance gate before creating a bundle. Collect, validate and
commit before choosing any further concurrency points.
