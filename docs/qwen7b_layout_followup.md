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
