# Scaling Studies

Related matrix entries: E032-E046.

Purpose: Run strong and weak scaling studies on 1, 2, and 4 H100 GPUs.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/scaling_studies/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
