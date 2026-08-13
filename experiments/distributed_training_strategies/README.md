# Distributed Training Strategies

Related matrix entries: E047-E062.

Purpose: Compare DDP, FSDP, and DeepSpeed ZeRO strategies.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/distributed/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
