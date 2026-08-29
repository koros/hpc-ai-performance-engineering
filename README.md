# Performance Engineering of Transformer Training and Inference on Multi-GPU H100 Systems

This repository contains the research artefacts, experiment framework, tracking documents, analysis outputs, and dashboard work for an MSc Artificial Intelligence dissertation at the University of Plymouth.

The project studies how transformer workloads behave on NVIDIA H100 systems in the Lovelace HPC facility, with selected NVIDIA L40S comparisons. Training is the core dissertation scope; inference is retained as a secondary high-impact scope because serving cost and latency often dominate long-term deployment.

## Current Experiment Plan

The latest plan is based on `Appendix_A_Comprehensive_Experiment_Matrix.docx` and uses a controlled factorial design across workload, GPU count, distributed strategy, precision, memory settings, data movement, inference framework, serving load, and hardware.

Matrix configs are normalized through a canonical phase map so every immutable result embeds validated `RQ1-RQ7` and `A01-A08` traceability identifiers alongside its source metadata.

Primary planning documents:

- [Current project status](docs/current_status.md): high-level snapshot of completed infrastructure, smoke-ready workloads, and remaining HPC submission tasks.
- [Experiment remediation and readiness guide](docs/experiment_remediation_guide.md): objective-aligned changes, acceptance criteria, and launch gates required before the full matrix is submitted.
- [Platform feasibility report](docs/platform_feasibility_report.md): platform-specific evidence boundaries and routing for Mac M1, L40/L40S, and H100 execution.
- [Enhanced platform experiment matrix](docs/platform_experiment_matrix.md): resolved, direct, local-dev, and deferred experiment routes across the three platforms.
- [L40S-first execution runbook](docs/platform_execution_runbook.md): gated waves, Slurm commands, evidence import, and H100 recovery workflow.
- [Measurement and trial protocol](docs/measurement_protocol.md): repetition, warm-up, timing, provenance, uncertainty, and outlier rules for research evidence.
- [Comprehensive experiment matrix](docs/experiment_matrix.md): full source-of-truth matrix with `E001` to `E200`, priority levels, metrics, and GPU-hour estimates.
- [Local dev runbook](docs/local_dev_runbook.md): Mac/local baseline-style runs for development while HPC access is unavailable.
- [Dev experiment matrix](docs/dev_experiment_matrix.md): rationale, parameters, and local-dev versus HPC-only experiment boundaries.
- [Generated experiment tracker](results/processed/experiment_tracker.md): current `E001-E200` matrix status generated from configs and raw results.
- [Editable experiment tracker](docs/experiment_tracker.md): planning notes for every planned experiment and analysis task.
- [HPC smoke runbook](docs/hpc_smoke_runbook.md): checklist for validating the cluster environment before full matrix submission.
- [HPC submission batches](docs/hpc_submission_batches.md): first environment-validation and baseline Slurm batches after smoke tests pass.
- [TensorRT-LLM E007 validation](docs/tensorrt_llm_validation.md): saved-engine contract, cluster setup, launch command, and activation criteria.
- [Experiment workspace](experiments/README.md): phase folders aligned with the dissertation proposal outline.
- [Experiment configs](configs/README.md): reusable config templates and concrete run configs.
- [Research result retention](results/README.md): importing, validating, committing, and backing up immutable HPC evidence.

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
python -m pip install -e ".[dev,yaml,workloads,distributed]"
```

Install vLLM separately only after confirming the cluster CUDA/PyTorch stack. Its wheels are CUDA-ABI-specific, so a
generic latest PyPI install can import a wheel that expects a different CUDA runtime. For example, if `torch.version.cuda`
reports `12.4` but `import vllm` fails looking for `libcudart.so.13`, remove that vLLM wheel and install or build one
matching the site runtime before running E006 or later vLLM rows.

Before running environment-validation waves from an allocated GPU node, check the active runtime:

```bash
python -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --require-deepspeed --require-vllm
```

DeepSpeed 0.17.x can still require `CUDA_HOME/bin/nvcc` during import-time optional-op probes. Use the project
`distributed` extra, which requires DeepSpeed 0.18.9 or newer, or load a full CUDA toolkit before running E005.

TensorRT-LLM, vLLM, and FP8/quantized model tooling remain vendor/backend-specific and are not silently substituted by
these extras.

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

The active path uses the tracked L40S platform profile while the H100 node is unavailable. Validate it, generate wave
0, and review the plan:

```bash
python3 -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
make render-l40s-wave PLATFORM_WAVE=0
make plan-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

This materializes generated configs under `build/platform_configs/l40s`, scripts under
`slurm/generated/l40s/wave_0`, and staged result JSON under `build/platform_results/l40s/wave_0`. The source
`E001-E200` configs remain unchanged. Rendering fails when a profile does not match the generated platform, preventing
H100/L40S result mislabelling.

Initialize status, dry-run, and then submit only after reviewing the generated files:

```bash
make init-platform-wave-status PLATFORM=l40s PLATFORM_WAVE=0
make dry-run-submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

Wave 0 uses one acceptance trial per condition. Core performance waves use three-task Slurm arrays based on the
tracked execution matrix. Each task writes an immutable, platform-qualified staged result with an explicit repetition
number; see the [measurement protocol](docs/measurement_protocol.md).

Import staged JSON through the evidence gate, then track accepted evidence separately from scheduler status:

```bash
make import-result-evidence SOURCE=build/platform_results/l40s/wave_0
make validate-result-evidence
make track-l40s-progress
```

When H100 returns, generate its independent conditions from the same logical matrix:

```bash
make render-h100-wave PLATFORM_WAVE=0
make track-h100-progress
```

Import JSON copied from the HPC or staged under `build/platform_results/...` through the integrity gate:

```bash
python3 -m hpc_ai_perf.cli import-result-evidence --source /tmp/hpc-result-staging
make validate-result-evidence
```

Detailed gated instructions are tracked in the [L40S-first execution runbook](docs/platform_execution_runbook.md).

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
make track-l40s-progress
make track-h100-progress
```

Validate experiment configs and raw results:

```bash
python -m hpc_ai_perf.cli validate
```

Require semantic research readiness (implemented workload, strategy, profiler, precision, and framework):

```bash
python -m hpc_ai_perf.cli validate --skip-results --research-ready
```

This stricter command must pass before deployment planning. Rows that require an external FP8/INT8/TensorRT artifact or an unsupported model/backend pairing are explicitly marked `execution_status: deferred`; they remain in the planning matrix but are skipped by bulk Slurm rendering until their backend contract is supplied. vLLM multi-GPU rows do not claim request-path NCCL telemetry that the adapter cannot expose.

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

The dashboard exposes Platform and Comparability filters. Its “All Platforms” view keeps chart series separate and
uses only additive counts or maxima for combined KPIs; statistical averages, confidence intervals, rankings, and
scaling baselines remain platform-local. `make check` includes mixed-platform Python and JavaScript regression tests.

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
