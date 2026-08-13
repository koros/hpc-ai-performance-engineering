# Precision Studies

Related matrix entries: E083-E096, E149-E200.

Purpose: Compare FP32, FP16, BF16, FP8, and selected quantized inference configurations.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete training precision configs live in `configs/precision_studies/`; inference precision configs live in `configs/inference/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
