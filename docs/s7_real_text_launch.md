# Fresh L40S S7: real-text batching and scaling

S7 has no recorded submission state on the HPC according to the user's check.
Start with cache preparation and fresh acceptance, then the original performance
wave. No historical S7 performance configs or runtime-fingerprinted files change.

## Scope

- Pinned GPT-2 Small and TinyLlama revisions and pinned WikiText-103 training split.
- Four modes: `pad_to_max`, `dynamic_pad`, `length_bucketed`, `pack`.
- Original performance IDs E638–E669: two workloads x four modes x
  1/2/4/8 GPUs x three repetitions = **96 trials**, 200 measured updates each.
- Fresh acceptance E810–E817: same eight input-mode risks as historical E370–E377,
  two GPUs, 10 warmup + 20 measured updates, one trial each = **8 trials**.
  Separate IDs avoid overlapping wave ownership and preserve historical canaries.
- `s7-real-text-canary` is L40S-only. `scaling-dd-real-text` on L40S now requires
  these eight current-runtime canaries, rather than unrelated FP8/FSDP canaries.
  Other performance-wave acceptance policies, including H100, remain unchanged.

The gate validates canonical checksums/config identity, current runtime, clean
source, update accounting, finite losses, memory, matching library versions,
mode identity, useful/processed/nominal token accounting, padding fraction,
rank imbalance and step-time tails. Packed inputs must have zero padding.
No canary is accepted merely because a different synthetic workload passed.

## Prepare on HPC (CPU only)

Push/pull the implementation first. From the main repo root, on the login host:

```bash
source slurm/modules_for_hpc_ai
source "$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo-recovered/bin/activate_recovered_vllm"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Permit downloads for cache preparation only.
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
python scripts/prefetch_scaling_deep_dive_assets.py --real-text-only --platform l40s

# Confirm that cached models/tokenizers/corpus can serve the largest S7 shapes.
python scripts/prefetch_scaling_deep_dive_assets.py --real-text-only --platform l40s --offline
```

This fetches only S7's two models, not all five deep-dive models. Corpus sizing
covers the full 200-update eight-GPU conditions, not just the short canaries.
The preflight tokenizes enough corpus to check both document and packed-token
budgets; it can take time and uses CPU, not a GPU allocation. Missing Python
packages, model files or dataset access must be resolved here before GPU jobs.

## Submit acceptance only

Wait for topology recovery to finish rather than overlapping full-node waves:

```bash
unset NCCL_ALGO NCCL_PROTO NCCL_P2P_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS CUDA_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/generate_s7_acceptance.py --check
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave s7-real-text-canary \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

The environment name contains vLLM but S7 runs the recovered PyTorch training
backend. No vLLM engine or Transformer Engine is required for these two shapes.

After completion:

```bash
python -m hpc_ai_perf.cli status-platform-wave \
  --platform-profile platforms/l40s.json --wave s7-real-text-canary
python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave s7-real-text-canary
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
python scripts/verify_s7_canaries.py
python -m hpc_ai_perf.cli stage-platform-wave \
  --platform-profile platforms/l40s.json --wave s7-real-text-canary
git diff --cached --stat
# Review other uncommitted L40S evidence included with the shared checksum manifest.
git commit -m "data: collect L40S S7 real-text canaries"
```

Review canary timings, memory, padding and accounting before the full allocation.
Then use the original wave, guarded by the focused acceptance gate:

```bash
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave scaling-dd-real-text \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

Use `scaling-dd-real-text` in the status/collect/stage commands after all 96
performance trials finish; verify evidence before staging and committing.

## Interpretation

Report useful-token throughput together with processed-token throughput,
padding fraction, rank-token imbalance, step-time p95, memory and GPU-hours.
The runtime's `throughput_tokens_sec` counts non-padding input tokens for real
text. It is not automatically the same denominator as synthetic token throughput
or the count of causal next-token prediction targets. Use the matching useful
work denominator when calculating real-text energy per token.

Padding modes use the same seeded document stream at fixed GPU count. Packing
joins documents into full windows and changes the samples/content consumed for
an update; it is a batching treatment, not proof of identical optimization or
convergence. Input digests encode the mode, so they must not be required to match
across modes. Compare scaling within a mode using its own one-GPU baseline.

Inputs are tokenized/preloaded before measurement: this study measures batching,
padding and rank-work effects, not end-to-end storage or DataLoader throughput.
