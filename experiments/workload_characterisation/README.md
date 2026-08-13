# Workload Characterisation

Related matrix entries: E020-E031.

Purpose: Profile compute intensity, memory pressure, data pipeline behaviour, and inference constraints for GPT-2 Small, DistilBERT, and TinyLlama before larger scaling and optimisation studies.

Expected artefacts in this folder include profiler run scripts, memory traces, data pipeline notes, generated config variants, and phase-specific analysis helpers. Concrete runnable configs live in `configs/workload_characterisation/`.

Canonical run metadata remains in `docs/experiment_matrix.md`. Current generated matrix status is written to `results/processed/experiment_tracker.md` after `python -m hpc_ai_perf.cli refresh`.
