# Experiment Remediation and Readiness Guide

## Purpose

This guide defines the changes required before the `E001-E200` matrix can produce research-valid evidence. It is organised around the seven research objectives in `docs/project_plan.md` and should be used as the implementation and launch checklist.

The current matrix is a useful planning artefact, but passing config validation does not mean that an experiment's declared strategy, profiler, precision, framework, or hardware is actually used. Full-matrix submission must remain paused until the launch gates at the end of this guide pass.

## Implementation Status (2026-08-18)

The code-level P0 repair is substantially complete. The runner now applies configured batch, accumulation, checkpointing, data-movement, distributed-strategy, precision, framework, serving-load, and profiler factors; multi-GPU training is rendered through `torchrun`; requested GPU models are checked; model revisions are pinned; matrix trials are preserved; and objective-specific analysis tables exclude failed runs.

Completed locally:

- Semantic capability validation is enforced by direct execution and Slurm rendering.
- CUDA/DDP/NCCL/FSDP/DeepSpeed/vLLM/TensorRT-LLM environment checks have executable adapters.
- DDP, FSDP full/sharded-grad, and DeepSpeed ZeRO-1/2/3 execution paths are wired.
- FP32, AMP FP16, and AMP BF16 paths report observed backends/dtypes; unsupported precision is rejected.
- Deterministic synthetic benchmark datasets and prompt tokens replace ambiguous `*_or_synthetic_*` labels.
- PyTorch and vLLM inference, real batch/load factors, DistilBERT concurrent streams, and multi-GPU data parallelism are implemented.
- nvidia-smi sampling, PyTorch trace export, DataLoader timings, and NCCL microbenchmarks are wired.
- Immutable trial files, config hashes, git/software/hardware provenance, measured GPU-hours, and actual useful-work counts are recorded.
- Trial, scaling, strategy, precision, memory, communication, data movement, inference, and hardware analysis tables are generated.

Deferred from the executable deployment scope or awaiting external evidence:

- `E086`, `E089`, `E093`, `E096`, `E151`, `E175`, and `E199` are marked deferred until a selected, validated FP8 backend and model path are supplied.
- `E152`, `E176`, and `E200` are marked deferred until a named quantization method, calibration protocol, and quantized artefact are supplied.
- `E007` now has a saved-engine build/load/request contract and is deferred only until a compatible TensorRT-LLM release succeeds on the target H100.
- `E131` and `E179` remain deferred until the validated E007 contract is integrated into the performance adapter; `E155` additionally requires a classification-compatible design.
- `E154` and `E155` are marked deferred because the declared generic causal-LM engines are not valid DistilBERT classification adapters.
- `E147-E148` and `E195-E196` no longer claim request-path NCCL metrics that the vLLM adapter cannot expose.
- Cluster execution must still prove CUDA/NCCL/Slurm behaviour, collect at least three independent core trials, and validate the measurement protocol on real H100/L40S nodes.
- A preregistered real-data quality protocol is still needed before making final RQ3 model-quality claims; current held-out checks primarily detect numerical instability.

## Priority Definitions

| Priority | Meaning |
| --- | --- |
| P0 | Launch blocker. Required before any matrix result is treated as dissertation evidence. |
| P1 | Required to answer a core research objective reliably. |
| P2 | Required for the stretch inference and hardware-comparison scope, or for stronger publication-quality evidence. |

## Research Traceability

| Research objective | Main research question(s) | Matrix coverage | Required evidence |
| --- | --- | --- | --- |
| Objective 1: Establish baselines | Foundation for all RQs | `E008-E019` | Repeated, steady-state single-GPU training and inference measurements with observed hardware, uncertainty, and provenance |
| Objective 2: Measure strong and weak scaling | RQ1 | `E032-E046` | Correct global-work accounting, multi-process execution, speedup, and separate strong/weak scaling efficiency |
| Objective 3: Compare distributed strategies | RQ2 | `E047-E062`, selected `E063-E082` | Verified DDP, FSDP, and ZeRO execution with comparable throughput and per-GPU memory measurements |
| Objective 4: Evaluate precision formats | RQ3 | `E083-E096`, selected `E149-E200` | Verified numerical backend, throughput, memory, stability, and task-quality results |
| Objective 5: Investigate communication and data movement | RQ4 | `E097-E120`, selected `E020-E030` | Real DataLoader transfers, collectives, synchronization timings, communication volume, and utilization traces |
| Objective 6: Develop the dashboard | RQ5 and RQ7 | `A01-A07` | Correct analysis tables, explicit bottleneck classification, uncertainty, actual GPU-hour/energy metrics, and RQ-specific views |
| Objective 7: Evaluate inference and H100 versus L40S | RQ6 and stretch inference scope | `E121-E200` | Verified serving engines and load, request-level distributions, controlled hardware placement, and observed platform metadata |

## Cross-Cutting Launch Blockers

These changes support every research objective and should be completed first.

### P0-01: Freeze the Authoritative Research Questions

- [x] Select one canonical list of research questions; `docs/project_plan.md` and `README.md` now use the same wording and numbering.
- [x] Treat useful work per GPU-hour and per Watt as formal RQ5.
- [x] Align the README and project plan wording and numbering.
- [x] Add validated RQ identifiers to every normalized matrix config's metadata through the canonical phase traceability map.
- [x] Add validated analysis-task identifiers where each phase supplies evidence for a planned synthesis.

Ready when: every experiment and analysis task traces to at least one agreed research objective or research question, and no document defines a conflicting set.

### P0-02: Add Semantic Configuration Validation

- [x] Validate that every `workload.name` is registered before Slurm scripts can be generated.
- [x] Validate profiler names against implemented profiler adapters.
- [x] Validate precision against the selected workload and backend.
- [x] Validate framework, distributed strategy, and hardware combinations against implemented capabilities.
- [x] Reject unknown workload and structural config parameters instead of silently accepting typos; declarative factor labels are retained as analysis metadata.
- [x] Validate `hardware.gpus == distributed.nodes * distributed.gpus_per_node`, except where an explicitly validated serving engine manages tensor parallelism.
- [x] Validate that `batch_size`, `micro_batch_size`, `per_gpu_batch_size`, and `global_batch_size` have unambiguous meanings and consistent values.
- [x] Validate Slurm CPU allocations against DataLoader workers and distributed process counts.
- [x] Add a `validate --research-ready` mode that reports every matrix config lacking an executable path.

Implementation areas:

- `src/hpc_ai_perf/validation/project.py`
- `src/hpc_ai_perf/config/experiment.py`
- `src/hpc_ai_perf/workloads/registry.py`
- tests covering every matrix experiment family

Ready when: the current unsupported environment-validation, TinyLlama-training, FP8, quantization, distributed, profiler, and serving-framework configurations fail preflight rather than consuming a scheduled job.

### P0-03: Implement Strict Environment Validation

- [x] Register an `environment_validation` workload and implement `E001-E007` checks.
- [x] Make GPU matrix jobs require CUDA; never permit automatic CPU fallback for an H100/L40S-labelled result.
- [x] Verify the allocated GPU count and map one process to each expected GPU.
- [x] Implement a real NCCL AllReduce check for 2- and 4-GPU jobs.
- [x] Implement one-step FSDP and DeepSpeed ZeRO checks.
- [x] Verify vLLM and TensorRT-LLM availability only when their corresponding experiments are selected.
- [x] Record observed GPU model, UUID, memory, driver, CUDA, PyTorch, NCCL, hostname, Slurm job ID, and visible-device count when exposed by the node.
- [x] Compare observed hardware with requested hardware and fail on a mismatch.

Ready when: `E001-E006` complete on Lovelace with all expected checks recorded as explicit result fields; optional `E007` must build, reload, and serve from a hashed engine artefact on its target GPU before activation.

### P0-04: Create a Reproducible Result and Trial Model

- [x] Stop overwriting the only JSON result for a matrix experiment ID.
- [x] Store each matrix attempt under a unique `run_id`.
- [x] Add fields for repetition number, warm-up count, measured iteration count, git commit, dirty-worktree state, config hash, model revision, dataset revision, command, and observed software/hardware.
- [x] Record actual elapsed GPU-seconds and GPU-hours, not only the planning estimate.
- [x] Record total useful samples/tokens so efficiency can be calculated dimensionally correctly.
- [x] Make distributed result writing rank-safe so only rank zero writes the aggregate record.
- [x] Update the schema, aggregation, tracker, and tests together; dashboard view work remains below.

Ready when: repeated runs are preserved, traceable to immutable code/config/model inputs, and cannot race or overwrite one another.

### P0-05: Standardise Measurement Practice

- [x] Apply `runtime.seed` to Python, PyTorch CPU, and every CUDA device.
- [x] Define deterministic settings and document where determinism is intentionally disabled for realistic performance.
- [x] Separate setup/model download from warm-up and measured steady-state regions.
- [x] Synchronize CUDA immediately before and after measured GPU regions.
- [x] Record CUDA-event duration alongside synchronized end-to-end wall time for transformer measured regions.
- [x] Define a minimum independent-trial count. Use at least three trials for core comparisons; five is preferable where queue budget permits.
- [x] Report mean, median, standard deviation, coefficient of variation, and confidence intervals where appropriate.
- [x] Define an outlier policy before collecting the final dataset.
- [x] Capture failures and OOMs as evidence without mixing their runtimes into completed-run averages.

Ready when: the same measurement protocol is used across comparable experiments and the analysis can quantify uncertainty.

## Objective 1: Establish Baseline Performance Measurements

Objective 1 requires a trustworthy single-GPU reference for every later comparison.

### Required Changes

- [x] Replace ambiguous datasets with fixed deterministic synthetic protocol names.
- [x] Label the main suite as controlled model-performance benchmarks and the DataLoader phase as CPU-to-GPU pipeline benchmarks.
- [x] Make the training adapters consume the configured batch-size field rather than falling back silently to batch size 1.
- [x] Record an explicit throughput unit for training tokens, input/classification tokens, generated tokens, and synthetic smoke work.
- [ ] Add validation loss or task-quality evaluation to the baseline protocol.
- [x] Add peak GPU memory, sampled utilization/power, measured GPU-hours, and estimated energy.
- [x] Render each retained baseline definition as a three-task Slurm array with explicit independent-trial numbers; legacy baseline/repeat rows remain separate matrix definitions.
- [x] Require the observed device to be the requested H100 before accepting a baseline result.
- [x] Implement the TinyLlama causal-training path while retaining it as optional scope.

### Acceptance Criteria

- [ ] GPT-2 Small and DistilBERT training and inference baselines complete on one observed H100.
- [ ] Each core baseline has the preregistered number of independent trials.
- [ ] Throughput, latency, loss/quality, memory, utilization, and actual GPU-hours are populated where applicable.
- [ ] Variance and confidence summaries can be reproduced from preserved per-trial records.
- [ ] Baseline reports contain no silent fallback, missing required metric, or ambiguous unit.

## Objective 2: Measure Strong and Weak Scaling

Objective 2 directly supports RQ1 and requires actual concurrent use of 1, 2, and 4 H100 GPUs.

### Required Changes

- [x] Render `torchrun` for multi-GPU DDP/FSDP/ZeRO configs.
- [x] Initialize and destroy the process group correctly for every distributed run.
- [x] Bind each local rank to one allocated GPU and record rank-to-device mapping.
- [x] Wrap the model in DDP for the GPT-2 Small and DistilBERT scaling studies.
- [x] Reduce timing with the slowest rank and calculate globally correct useful-work throughput.
- [ ] For strong scaling, keep the total global workload fixed and document how batches that do not divide evenly are handled.
- [ ] For weak scaling, keep work per GPU fixed and increase the global workload in proportion to GPU count.
- [x] Calculate strong-scaling speedup and efficiency separately from weak-scaling efficiency.
- [x] Base scaling calculations on the measured steady-state region, not model-loading or download time.
- [x] Keep DDP constant across the TinyLlama 1/2/4-GPU scaling curve.

### Acceptance Criteria

- [ ] Runtime evidence shows world sizes 1, 2, and 4 and all requested GPUs active.
- [ ] Global token/sample counts match the strong- or weak-scaling protocol exactly.
- [ ] Scaling tables use measured steady-state runtime and globally aggregated throughput.
- [ ] Strong- and weak-scaling formulas have unit tests using ideal and deliberately inefficient synthetic cases.
- [ ] Scaling plots include uncertainty across independent trials.

## Objective 3: Compare Distributed Training Strategies

Objective 3 supports RQ2: selecting the best throughput-to-memory trade-off among DDP, FSDP, and DeepSpeed ZeRO.

### Required Changes

- [x] Implement DDP, FSDP full shard, FSDP sharded-grad, and ZeRO-1/2/3 as separate execution paths.
- [x] Make `fsdp_config`, `deepspeed_config`, and strategy-specific config fields control the actual runtime.
- [ ] Pin compatible PyTorch, DeepSpeed, CUDA, and Transformer versions.
- [ ] Use identical model revision, sequence length, effective global batch, optimizer, step count, data, and precision for strategy comparisons.
- [x] Implement gradient accumulation and include it in effective-global-batch accounting.
- [ ] Measure peak allocated and reserved memory per rank, optimizer/model/gradient state where feasible, and aggregate max memory across ranks.
- [ ] Measure throughput, step time, communication time, synchronization/idle time, and OOM behavior.
- [ ] Verify numerical equivalence within a documented tolerance using loss trajectories or a short convergence check.
- [ ] Separate capacity experiments from equal-work performance comparisons.

### Acceptance Criteria

- [ ] Raw results state the observed strategy and strategy settings, not only the requested label.
- [ ] DDP, FSDP, and supported ZeRO stages complete the same controlled workload on two GPUs.
- [ ] Selected DDP/FSDP comparisons complete on four GPUs.
- [ ] The analysis can plot throughput against maximum per-GPU memory with uncertainty and identify dominated configurations.

## Objective 4: Evaluate Precision Formats

Objective 4 supports RQ3 and must measure both speed and useful model quality.

### Required Changes

- [x] Report the observed precision backend, compute dtype, and parameter dtype.
- [x] Implement FP32 as the reference path.
- [x] Implement FP16 training with autocast and gradient scaling.
- [x] Implement BF16 with autocast/mixed-precision strategy support.
- [x] Reject FP8 at preflight until a supported H100 backend is selected.
- [x] Reject INT8/4-bit as ordinary `torch.dtype` substitutions; they require quantized artefacts.
- [ ] Pin and record quantization method, calibration procedure, backend, model revision, and quantized artefact.
- [x] Use the same deterministic inputs and seeds across precision comparisons.
- [ ] Define workload-appropriate quality metrics: language-model loss/perplexity for GPT-2, classification accuracy/F1 for DistilBERT, and a documented inference-quality check for generative models.
- [ ] Record NaNs, overflows, divergence, loss stability, memory, throughput, latency, and energy.

### Acceptance Criteria

- [x] Unsupported precisions fail semantic preflight rather than starting a GPU job.
- [ ] Every completed precision result records the observed backend and compute/parameter dtype.
- [ ] Performance comparisons include memory and quality change relative to FP32 or the declared reference.
- [ ] The precision report identifies Pareto trade-offs instead of ranking throughput alone.

## Objective 5: Investigate Communication and Data Movement Bottlenecks

Objective 5 supports RQ4 and requires real communication and I/O paths.

### Required Changes

- [x] Add a profiler registry that constructs the profilers named in each config.
- [x] Implement GPU utilization, memory, clock, temperature, and power sampling through `nvidia-smi`.
- [x] Implement PyTorch CPU/CUDA/memory trace export; bounded schedules still need cluster tuning.
- [x] Collect standalone NCCL collective duration, payload bytes, and effective bandwidth.
- [ ] Measure AllReduce/FSDP exchange time, synchronization time, communication-to-compute ratio, and GPU idle time.
- [ ] Collect NVLink counters where supported and document the fallback when counters are unavailable.
- [x] Implement a deterministic Dataset and DataLoader path with configurable workers, pinned memory, and prefetch factor.
- [x] Time data loading and host-to-device transfer separately from the measured optimizer step.
- [x] Raise Slurm CPU allocation to cover configured workers and distributed ranks.
- [ ] Control cache state and distinguish cold-cache from warm-cache storage tests.
- [x] Preserve rank-specific PyTorch trace artefacts and link them from profiler metrics.

### Acceptance Criteria

- [ ] Changing DataLoader parameters changes the instantiated DataLoader and is visible in observed runtime metadata.
- [ ] Communication configs execute real multi-rank collectives and return non-empty communication metrics.
- [x] Profiler-enabled configs fail if required metrics cannot be collected; they cannot silently complete with empty profiler output.
- [ ] The analysis can attribute measured time among compute, communication, synchronization, data loading, and transfer with documented limitations.

## Objective 6: Develop a Performance Visualisation Dashboard

Objective 6 supports RQ5 and RQ7 and should turn valid measurements into reproducible conclusions.

### Required Changes

- [x] Exclude failed runs from performance means/rankings while reporting failed-trial counts separately.
- [x] Calculate actual-GPU-hour efficiency from measured useful work and GPU time.
- [x] Add energy efficiency only from sampled power/energy fields.
- [x] Add separate trial, scaling, strategy, memory, precision, communication, data-movement, inference, and hardware tables.
- [ ] Implement an explicit bottleneck-classification method with documented features, thresholds, confidence, and an `insufficient_data` outcome.
- [x] Carry explicit throughput units into analysis tables and dashboard filters; mixed-unit dashboard views do not report a single best-throughput value.
- [x] Include trial counts, confidence intervals, and explicit insufficient-data warnings in the dashboard trial-evidence view; plot error bars remain future presentation work.
- [ ] Add RQ-specific dashboard views and a trace from each conclusion to the underlying trial records.
- [x] Add automated integrity checks that block refresh on invalid evidence and mark missing cluster evidence explicitly in the readiness report.

### Acceptance Criteria

- [ ] Each core research question has a dedicated analysis table and dashboard view.
- [ ] Bottleneck classifications are reproducible from stored metrics and can state that evidence is insufficient.
- [ ] Cost and energy figures use measured quantities with explicit units.
- [ ] Dashboard filters and rankings cannot silently mix failed runs, dev results, hardware platforms, workload units, or incompatible modes.

## Objective 7: Evaluate Inference and Compare H100 with L40S

Objective 7 is a stretch goal, but its Must Have inference entries still require valid execution if retained in the submitted matrix.

### Required Changes

- [x] Route `pytorch_eager` and `vllm` to different adapters; supply E007 with a saved-engine validation contract while keeping performance rows deferred until adapter integration.
- [x] Reject incompatible DistilBERT/vLLM/TensorRT-LLM combinations and route valid classification sweeps through PyTorch.
- [x] Generate deterministic exact-length token inputs within each model's supported context window.
- [x] Make batch size and concurrent requests control actual PyTorch streams or vLLM active request groups.
- [x] Implement request warm-up and repeated p50/p95 latency measurement.
- [x] Measure vLLM TTFT when the engine exposes request timestamps.
- [x] Report generated-token throughput for causal LMs and input-token/sample throughput for classification.
- [x] Pass validated tensor-parallel size to vLLM and data-parallel size to DistilBERT PyTorch inference.
- [x] Create separate H100 and L40S profile templates and reject profile/config platform mismatches.
- [x] Fail results when the observed GPU model does not match the experiment label.
- [ ] Control software versions, model revision, power mode, warm-up, batch/load, and measurement duration across hardware comparisons.
- [ ] Report absolute performance, per-GPU efficiency, memory capacity, energy, and any price/cost assumptions separately.

### Acceptance Criteria

- [ ] Raw results identify the serving engine actually used.
- [ ] Latency percentiles are calculated from a preserved request-level distribution rather than one request duplicated into p50 and p95.
- [ ] Concurrency, batching, prompt length, and tensor-parallel sweeps show observed runtime settings matching their configs.
- [ ] H100-labelled and L40S-labelled results record the corresponding observed devices and use the same controlled workload protocol.

## Recommended Implementation Order

| Stage | Work | Exit condition |
| --- | --- | --- |
| 0 | P0-01 research-question freeze | One authoritative objective/RQ mapping exists |
| 1 | P0-02 through P0-05 framework repairs | Semantic validation, strict device checks, preserved trials, and valid timing are tested |
| 2 | Environment validation and Objective 1 | `E001-E006` pass and repeated GPT-2/DistilBERT H100 baselines are trustworthy |
| 3 | Objective 2 | One GPT-2 1/2/4-GPU DDP pilot passes all scaling integrity checks |
| 4 | Objectives 3-5 | Distributed, precision, memory, profiler, communication, and DataLoader factors are genuinely applied |
| 5 | Objective 6 | Correct RQ-specific analysis and missing-evidence gates are operational |
| 6 | Objective 7 | Serving adapters and H100/L40S placement pass their own pilots before the stretch matrix is released |

## Full-Matrix Launch Gates

Do not submit all Must Have experiments until every applicable gate below passes.

- [ ] Research questions and metrics are consistent across the README, project plan, matrix, and analysis plan.
- [ ] `validate --research-ready` reports zero issues for the selected batch.
- [ ] `E001-E006` pass on Lovelace and record requested-versus-observed environment checks.
- [x] No H100/L40S experiment can fall back to CPU.
- [ ] The baseline pilot meets the repetition, timing, provenance, and required-metric protocol.
- [ ] A 1/2/4-GPU pilot proves all allocated GPUs are active and global work is accounted for correctly.
- [x] Every enabled profiler returns its required metrics or fails the experiment.
- [ ] Every varied config parameter is asserted against observed runtime state.
- [x] Failed/OOM jobs are separated from completed-run performance statistics.
- [x] Previous trials are preserved and distributed ranks cannot overwrite result files.
- [ ] Analysis unit tests cover strong scaling, weak scaling, actual GPU-hours, percentiles, uncertainty, and missing data.
- [ ] H100 and L40S jobs use hardware-specific scheduling and verify the allocated GPU model.

## First Safe Pilot After P0 Repairs

Run the following sequence before releasing a larger batch:

1. Run `E001`, then `E002`, `E003`, and `E004`; run `E005-E006` only after their dependency preflight passes.
2. Run repeated one-H100 GPT-2 Small and DistilBERT baselines using the new trial model.
3. Review observed device, throughput units, memory, utilization, timing region, and variability.
4. Run the GPT-2 Small DDP strong-scaling series `E032-E034`.
5. Confirm world size, rank mapping, global tokens, GPU activity, communication metrics, speedup, and efficiency manually.
6. Release the next objective-aligned batch only after the pilot evidence passes its acceptance criteria.

Local synthetic and tiny-model runs may continue during remediation, but they are software-path checks and must not be reported as H100/L40S research evidence.
