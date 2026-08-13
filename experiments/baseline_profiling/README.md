# Baseline Profiling

Related matrix entries: E008-E019.

Purpose: Establish repeatable one-GPU training and inference baselines and variance estimates.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/baseline_profiling/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
