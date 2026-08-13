# Environment Validation

Related matrix entries: E001-E007.

Purpose: Validate CUDA, PyTorch, NCCL, FSDP, DeepSpeed, vLLM, and optional TensorRT-LLM before collecting experimental results.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/environment_validation/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
