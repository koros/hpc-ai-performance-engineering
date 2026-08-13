# Memory Optimisation

Related matrix entries: E063-E082.

Purpose: Evaluate activation checkpointing, micro-batch size, and gradient accumulation trade-offs.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/memory_optimisation/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
