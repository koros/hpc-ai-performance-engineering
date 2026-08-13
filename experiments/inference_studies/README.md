# Inference Studies

Related matrix entries: E129-E200.

Purpose: Evaluate PyTorch eager, vLLM, TensorRT-LLM, batching, concurrency, prompt length, tensor parallelism, and inference precision.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/inference/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
