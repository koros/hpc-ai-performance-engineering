# Performance Engineering of Transformer Training and Inference on Multi-GPU H100 Systems

This repository contains the research artefacts, experiment framework, tracking documents, analysis outputs, and dashboard work for an MSc Artificial Intelligence dissertation at the University of Plymouth.

The project studies how transformer workloads behave on NVIDIA H100 systems in the Lovelace HPC facility, with selected NVIDIA L40S comparisons. Training is the core dissertation scope; inference is retained as a secondary high-impact scope because serving cost and latency often dominate long-term deployment.

## Current Experiment Plan

The latest plan is based on `Appendix_A_Comprehensive_Experiment_Matrix.docx` and uses a controlled factorial design across workload, GPU count, distributed strategy, precision, memory settings, data movement, inference framework, serving load, and hardware.

Matrix configs are normalized through a canonical phase map so every immutable result embeds validated `RQ1-RQ7` and `A01-A08` traceability identifiers alongside its source metadata.

Primary planning documents:

- [Current project status](docs/current_status.md): high-level snapshot of completed infrastructure, smoke-ready workloads, and remaining HPC submission tasks.
- [Experiment remediation and readiness guide](docs/experiment_remediation_guide.md): objective-aligned changes, acceptance criteria, and launch gates required before the full matrix is submitted.
- [Measurement and trial protocol](docs/measurement_protocol.md): repetition, warm-up, timing, provenance, uncertainty, and outlier rules for research evidence.
- [Comprehensive experiment matrix](docs/experiment_matrix.md): full source-of-truth matrix with `E001` to `E200`, priority levels, metrics, and GPU-hour estimates.
- [Local dev runbook](docs/local_dev_runbook.md): Mac/local baseline-style runs for development while HPC access is unavailable.
- [Dev experiment matrix](docs/dev_experiment_matrix.md): rationale, parameters, and local-dev versus HPC-only experiment boundaries.
- [Generated experiment tracker](results/processed/experiment_tracker.md): current `E001-E200` matrix status generated from configs and raw results.
- [Editable experiment tracker](docs/experiment_tracker.md): planning notes for every planned experiment and analysis task.
- [HPC smoke runbook](docs/hpc_smoke_runbook.md): checklist for validating the cluster environment before full matrix submission.
- [HPC submission batches](docs/hpc_submission_batches.md): first environment-validation and baseline Slurm batches after smoke tests pass.
- [Experiment workspace](experiments/README.md): phase folders aligned with the dissertation proposal outline.
- [Experiment configs](configs/README.md): reusable config templates and concrete run configs.

## Proposal Outline

The repository now mirrors the project proposal structure:

| Proposal section | Matrix coverage |
| --- | --- |
| Environment validation | `E001-E007` |
| Benchmark selection | GPT-2 Small, DistilBERT, TinyLlama |
| Baseline profiling | `E008-E019` |
| Precision studies | `E083-E096`, selected inference precision in `E149-E200` |
| Scaling studies | `E032-E046` strong scaling and `E035-E043` weak scaling |
| Distributed training strategies | `E047-E062` DDP, FSDP, ZeRO-1, ZeRO-2, ZeRO-3 |
| Memory optimisation | `E063-E082` |
| Data movement and communication | `E097-E120` |
| Hardware comparison | `E121-E128` H100 vs L40S |
| Inference studies | `E129-E200`, subject to time and platform stability |
| Integrated analysis and dashboard | `A01-A07` |
| Recommendations | `A08` |

## Research Questions

1. How efficiently do transformer workloads scale from one to four H100 GPUs?
2. Which distributed training strategy provides the best throughput-to-memory trade-off?
3. What impact do FP32, FP16, BF16, FP8, and selected quantized inference formats have on performance and quality?
4. How significant are communication, synchronisation, and data movement bottlenecks?
5. Which configurations produce the best useful work per GPU-hour and per Watt?
6. How do H100 and L40S platforms compare for selected AI workloads?
7. Can experiment results be integrated into a dashboard that identifies bottlenecks and supports practical recommendations?

## Workloads

| Workload | Type | Primary use |
| --- | --- | --- |
| GPT-2 Small | Decoder transformer | Training and inference baseline |
| DistilBERT | Encoder transformer | Training and inference comparison |
| TinyLlama | Small modern LLM | Primarily inference; optional training |

Current adapter implementation status is tracked in [workload catalog](docs/workload_catalog.md).

## Priority and Budget

| Priority | Meaning | Estimated GPU-hours |
| --- | --- | --- |
| Must Have | Required for a complete dissertation | 89.4 |
| Should Have | Important for stronger impact | 46.1 |
| Optional | Stretch goal | 12.1 |
| Total | Full planned matrix | 147.6 |

The reduced feasible scope in the appendix is approximately 95-118 GPU-hours, prioritising environment validation, GPT-2 and DistilBERT baselines, scaling, distributed strategy comparison, precision and memory studies, data movement, communication analysis, and a focused inference subset.

## Repository Structure

```text
hpc-ai-performance-engineering/
├── configs/                  # JSON/YAML experiment configurations
│   ├── baseline_profiling/
│   ├── communication_analysis/
│   ├── data_movement/
│   ├── dev/
│   ├── distributed/
│   ├── environment_validation/
│   ├── hardware_comparison/
│   ├── inference/
│   ├── local/
│   ├── memory_optimisation/
│   ├── precision_studies/
│   ├── scaling_studies/
│   ├── templates/
│   ├── training/
│   └── workload_characterisation/
├── dashboard/                # Static dashboard for processed results
├── docs/                     # Project plan, matrix, tracker, architecture, hardware notes
├── experiments/              # Phase folders aligned with the proposal outline
├── notebooks/                # Exploratory analysis notebooks
├── results/                  # Raw results, processed outputs, and figures
├── schemas/                  # Result schema definitions
├── scripts/                  # Utility scripts
├── slurm/                    # Slurm scripts, profiles, generated plans, and smoke presets
├── src/hpc_ai_perf/          # Experiment config, runner, profiling, results, workload code
├── tests/                    # Local validation tests
└── workloads/                # Workload-specific assets and notes
```

## Metrics

The matrix tracks runtime, samples/sec, tokens/sec, loss, GPU-hours, GPU utilisation, GPU memory, CPU utilisation, data-loading time, speedup, scaling efficiency, NCCL bandwidth, AllReduce time, synchronisation overhead, NVLink utilisation, tokens/sec, time-to-first-token, p50 and p95 latency, power draw, energy consumption, throughput/GPU-hour, and throughput/Watt.

## Local Validation

Install the package in editable mode and run the smoke tests:

```bash
python -m pip install -e ".[dev,yaml]"
pytest
```

Install optional transformer workload dependencies when moving beyond the synthetic smoke path:

```bash
python -m pip install -e ".[dev,yaml,workloads]"
```

On the Linux GPU cluster, install the extras required by the selected batch:

```bash
python -m pip install -e ".[dev,yaml,workloads,distributed,serving]"
```

TensorRT-LLM and FP8/quantized model tooling remain vendor/backend-specific and are not silently substituted by these extras.

Run the optional GPT-2 Small smoke workload after installing those dependencies:

```bash
python -m hpc_ai_perf.cli run --config configs/local/gpt2_small_smoke.json
```

Run the optional DistilBERT smoke workload:

```bash
python -m hpc_ai_perf.cli run --config configs/local/distilbert_smoke.json
```

Run the optional TinyLlama-style inference smoke workload:

```bash
python -m hpc_ai_perf.cli run --config configs/local/tinyllama_inference_smoke.json
```

Review the Slurm smoke presets for running these checks on an HPC GPU node:

```bash
make list-slurm-smoke
```

Run the local synthetic smoke benchmark:

```bash
python -m hpc_ai_perf.cli run --config configs/local/synthetic_smoke.json
```

Run local dev experiments on a Mac or workstation while HPC is unavailable:

```bash
make run-dev-refresh
```

These `dev_` runs are workflow checks and small local sweeps, not H100 measurements. Their progress is tracked separately in `results/processed/dev_experiment_tracker.md`.

List matrix experiment configs:

```bash
python -m hpc_ai_perf.cli list-configs
```

Render a Slurm submission script for a config:

```bash
python -m hpc_ai_perf.cli render-slurm --config configs/environment_validation/E001_cuda_pytorch_1gpu.json
```

Render Slurm scripts for a filtered group:

```bash
python -m hpc_ai_perf.cli render-slurm-bulk --phase "Environment validation" --priority "Must Have"
```

## HPC Submission Workflow

Prepare a local Slurm profile from the tracked example and edit it for the target cluster:

```bash
cp slurm/profiles/example.json slurm/profiles/cluster.local.json
```

Use `slurm/profiles/example_l40s.json` as the starting point for L40S jobs. Rendering fails when a profile platform does not match the config, preventing H100/L40S result mislabelling.

Generate the first environment-validation and baseline batches with site-specific settings:

```bash
SLURM_PROFILE=slurm/profiles/cluster.local.json make check-slurm-profile
SLURM_PROFILE=slurm/profiles/cluster.local.json make plan-slurm-first-batches
```

The baseline scripts are three-task Slurm arrays. Each task writes an immutable trial with an explicit repetition number; see the [measurement protocol](docs/measurement_protocol.md).

Initialize job-status CSVs before submitting jobs:

```bash
make init-slurm-first-status
make dry-run-submit-slurm-first
```

On the cluster, submit a batch and record returned Slurm job IDs automatically:

```bash
python -m hpc_ai_perf.cli submit-slurm-batch \
  --script-dir slurm/generated/environment_validation_must_have \
  --status results/processed/slurm_environment_validation_status.csv
```

Or after a manual `sbatch` command, record the returned Slurm job ID:

```bash
python -m hpc_ai_perf.cli record-slurm-job \
  --status results/processed/slurm_environment_validation_status.csv \
  --experiment-id E001 \
  --sbatch-output "Submitted batch job 123456"
```

Generate Markdown status reports:

```bash
make report-slurm-first-status
```

Detailed submission instructions are tracked in [HPC submission batches](docs/hpc_submission_batches.md).

Aggregate raw JSON results into a processed experiment table:

```bash
python -m hpc_ai_perf.cli aggregate-results
```

Build analysis tables for trials, phase summaries, rankings, scaling, distributed strategies, precision, memory, communication, data movement, inference, and hardware comparisons:

```bash
python -m hpc_ai_perf.cli analyze-results
```

Generate progress tracker reports from configs and raw results:

```bash
python -m hpc_ai_perf.cli track-progress
make track-dev-progress
```

Validate experiment configs and raw results:

```bash
python -m hpc_ai_perf.cli validate
```

Require semantic research readiness (implemented workload, strategy, profiler, precision, and framework):

```bash
python -m hpc_ai_perf.cli validate --skip-results --research-ready
```

This stricter command is expected to remain non-zero for explicitly unsupported FP8/INT8/TensorRT artefact rows, incompatible DistilBERT causal-engine comparisons, and vLLM multi-GPU rows that request unavailable request-path NCCL telemetry. Those rows cannot be rendered or run until their reported blocker is resolved; valid objective-aligned batches can be rendered independently.

Refresh validation, aggregation, analysis, tracker outputs, and readiness reporting in one pass:

```bash
python -m hpc_ai_perf.cli refresh
```

Generate an HPC submission readiness report:

```bash
python -m hpc_ai_perf.cli readiness-report
```

Serve the static dashboard:

```bash
make serve-dashboard
```

## Expected Outputs

- Reproducible experiment suite and configuration catalogue
- Baseline, scaling, precision, memory, communication, hardware, and inference results
- Integrated analysis and dashboard
- Dissertation figures and tables
- Practical recommendations for efficient transformer training and serving on modern GPU systems

## Author

Geoffrey Koros
MSc Artificial Intelligence
University of Plymouth

## License

To be determined.
