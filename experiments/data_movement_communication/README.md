# Data Movement and Communication

Related matrix entries: E097-E120.

Purpose: Study DataLoader settings, pinned memory, prefetching, NCCL, AllReduce, and synchronisation overhead.

Expected artefacts in this folder include run scripts, notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/data_movement/` and `configs/communication_analysis/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
