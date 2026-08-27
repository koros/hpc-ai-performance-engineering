# Measurement and Trial Protocol

This protocol applies to dissertation evidence collected from the `E001-E200` matrix. Local and synthetic runs are software-path checks and are not H100/L40S evidence.

## Trial Identity and Repetition

- A trial is one independent process launch of one unchanged experiment config.
- Core comparisons require at least three completed trials. Five are preferred when queue and GPU-hour budgets permit.
- `--repetitions N` renders a Slurm job array with task numbers `1-N`. The baseline Must Have Make target uses three repetitions.
- Each raw result records `repetition_number` and includes it in the unique `run_id`. Failed and OOM trials are preserved and counted but are not included in performance statistics.
- Platform evidence also records `condition_id` (`E###@l40s` or `E###@h100`) and `comparability`. The logical E-number alone is not sufficient to identify a trial.
- Repetitions do not modify the configured seed. This keeps workload inputs fixed while measuring run-to-run system variation. Configs that intentionally vary seeds remain separate experiment definitions.

## Setup, Warm-up, and Measurement

Model construction, dependency checks, model loading, and dataset preparation occur before the measured region. Transformer adapters then run an unmeasured warm-up followed by the configured measured iterations.

When `workload.parameters.warmup_steps` is absent, the default is zero for a one-iteration smoke check and otherwise 10% of measured iterations, bounded to one through five warm-up iterations. Raw results record both `warmup_iterations` and `measured_iterations`.

CUDA is synchronized at measured-region boundaries and after measured steps before wall-clock timings are read. Transformer adapters also record `cuda_event_seconds` across the measured device region. `measured_seconds` is the comparable steady-state end-to-end duration; `runtime_seconds` remains the full process duration and must not replace it in scaling calculations. A region-level CUDA event does not by itself provide kernel-by-kernel or communication/compute attribution.

Every result also records `throughput_unit`: `training_tokens_per_second` for optimizer work, `generated_tokens_per_second` for causal serving, `input_tokens_per_second` for classification inference, or `synthetic_tokens_per_second` for local smoke work. Rankings and charts must be filtered or grouped by this field; unlike units are not interchangeable.

## Reproducibility and Determinism

Every run applies `runtime.seed` to Python, PyTorch CPU, and CUDA devices. `runtime.deterministic` defaults to `false`: deterministic inputs and seeds are retained, while deterministic PyTorch algorithms are disabled and cuDNN autotuning is enabled for representative performance. Set it to `true` for numerical-reproducibility diagnostics; those runs must not be mixed with default performance trials.

Completed matrix evidence records the config hash, git commit and clean/dirty state, command, repetition, model revision, versioned dataset protocol, software versions, and observed hardware. Validation rejects completed matrix records with missing provenance, a dirty worktree, or a config hash that does not match the embedded config.

Platform-resolved evidence additionally records the logical source-config SHA-256, generated execution platform,
wave, required-trial count, result root, and comparability class. L40S and H100 trials are stored and aggregated as
separate conditions even when they originate from the same logical E-number.

## Statistics and Missing Evidence

The trial summary reports completed and failed counts, mean, median, sample standard deviation, coefficient of variation, and a two-sided 95% Student's t confidence interval. Fewer than three completed trials are labelled `insufficient_data`.

No trial is automatically excluded as an outlier. Before final collection, investigate extreme observations using scheduler logs, profiler traces, thermals, clocks, and contention. Exclude a trial only for a documented protocol violation or measurement failure, preserve the raw record, and report results with and without the exclusion. The generated summary therefore reports `outliers_excluded=0`.

## Pilot Acceptance

Before a larger batch is released:

1. All required environment-validation jobs must complete on the requested Lovelace hardware.
2. Each core baseline must have at least three valid trials and a non-empty confidence interval.
3. Requested and observed GPU model, world size, strategy, precision/backend, batch semantics, useful-work counts, and profiler outputs must agree.
4. Variance must be investigated when the coefficient of variation is unexpectedly high; no universal cutoff is imposed before pilot evidence establishes normal cluster variability.
5. Scaling and strategy batches may proceed only after the single-GPU baseline protocol passes.
