# Column and statistical conventions

## Trial and summary identifiers

- `source_record`: path relative to `results/raw`; unique source-file identifier.
- `source_sha256`: SHA-256 of that immutable JSON file.
- `platform`, `run_id`: use together when joining trial/sample tables.
- `experiment_id`: E-number. It is not an independent repetition identifier.
- `submission_cohort_id`: config hash plus recorded runtime identity. Bundle-local
  config differences deliberately separate submission cohorts.
- `runtime_cohort_id`: available package/driver versions and runtime fingerprints;
  source commit is used as a fallback when fingerprints are absent.
- `selected_current_cohort`: whether this is the latest attempt cohort.
- `analysis_eligible`: selected, completed, error-free and not explicitly dirty.
  Also filter `role == "performance"` for benchmark summaries.
- `role`: performance, canary or diagnostic, inferred from the recorded config.
- `comparability`: original evidence comparability classification.
- `study_design_note`, `motivating_experiment_ids`, `follow_up_planned_on`,
  `planned_after_results_commit`: exploratory follow-up provenance. ID lists are
  semicolon-delimited strings in both formats, keeping records flat.

## Experimental factors

`model`, `model_revision`, `dataset_revision`, `platform`, `gpu_model`, `gpus`,
`precision`, `model_load_dtype`, `precision_backend`, `strategy`, `strategy_detail`,
`serving_layout`, `tensor_parallel_size`, `replica_count`, `per_gpu_batch_size`,
`global_batch_size`, `sequence_length`, `gradient_accumulation_steps`, `prompt_tokens`,
`output_tokens_requested`, `concurrency`, DDP settings and input protocol are separate
columns. Do not aggregate across factors accidentally. Empty values mean not
recorded/not applicable, not an assumed default.

## Metrics

| Column | Unit / meaning |
|---|---|
| `throughput_tokens_sec` | Recorded tokens/second; interpret using `mode` and `throughput_unit` |
| `energy_j_measured_region` | Joules integrated over the measured region across the profiled GPUs |
| `joules_per_output_token` | Measured joules / generated output tokens, within an inference trial |
| `joules_per_processed_token` | Measured joules / recorded processed tokens, within a training trial |
| `measured_seconds` | Workload measured interval; may differ from full runtime |
| `runtime_seconds` | Recorded overall trial duration; not interchangeable with measured duration |
| `peak_memory_gib` | Recorded memory metric in GiB; device/allocator scope depends on the source profiler |
| `ttft_p95_ms` | Within-trial p95 backend time to first token, milliseconds |
| `time_per_output_token_p95_ms` | Within-trial p95 decode time/output token, milliseconds |
| `latency_p95_ms` | Within-trial p95 recorded request latency, milliseconds |
| `requests`, `generated_tokens`, `processed_tokens` | Per-trial work counts, where available |

In generic condition summaries each metric has `_mean`, `_sd`, `_n`, `_ci95_low`
and `_ci95_high`. `trial_count` counts all included repetitions; `_n` may be smaller
if a particular metric is missing. Intervals are approximate and reflect trial
variation, not systematic measurement uncertainty. Do not clip a negative lower
CI to zero without stating that plotting choice.

## Training-specific tables

These reuse the controlled deep-dive analysis and its matching/contrast rules,
restricted to selected cohorts and separated by runtime identity. Their numeric
columns remain numbers in JSON. Some original column names use `_gb` although
this project reports those memory figures in GiB.

`scaling_efficiency` is a fraction (1 = ideal); multiply by 100 for percent.
`baseline_experiment_id` identifies the matched one-GPU reference. Missing baselines
leave efficiency null. `scaling_basis = matched_per_gpu_workload` explicitly means
batch, sequence and accumulation are matched per GPU. Do not label these curves
fixed-global-batch strong scaling merely because a source config says `strong`:
inspect `global_batch_size` as GPU count changes. `scaling_protocol_recorded` in the
trial tables preserves that original label without reinterpreting it.

`throughput_change_pct` is the controlled effect in percent.
`efficiency_change_points` is percentage points. Effect confidence intervals and
verdicts use the existing deep-dive method: 3% materiality and a 95% interval
excluding zero. These are exploratory contrasts, not multiple-testing-adjusted
confirmatory results. Never compare different models using a same-model baseline.

## Request samples

`metric`, `value`, `unit`, `sample_index` form a long table. Current timing samples
are in seconds. Use `value * 1000` for milliseconds. Order is the source array order;
it is not a globally unique request identifier or an arrival trace. Preserve
`platform`, `run_id` and `runtime_cohort_id` when grouping or bootstrapping.

## Coverage

`policy_status` preserves the matrix policy; `coverage_status` describes collected
latest-cohort evidence. They are separate concepts. The required count applies to
one platform condition. A complete condition on L40S does not imply H100 completion.
