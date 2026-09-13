# Performance Engineering of Transformer Training and Inference on Multi-GPU H100 Systems

This repository contains the research artefacts, experiment framework, tracking documents, analysis outputs, and dashboard work for an MSc Artificial Intelligence dissertation at the University of Plymouth.

The project studies how transformer workloads behave on NVIDIA H100 systems in the Lovelace HPC facility, with selected NVIDIA L40S comparisons. Training is the core dissertation scope; inference is retained as a secondary high-impact scope because serving cost and latency often dominate long-term deployment.

The evolving experiment evidence is available in the [public research dashboard](https://koros.github.io/hpc-ai-performance-engineering-dashboard/). The dashboard is published from a sanitized derived-data artifact while this complete research repository is prepared for open-source release.

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
- [GPU telemetry revalidation](docs/gpu_telemetry_revalidation.md): UUID-bound device attribution, legacy evidence treatment, telemetry audit, and cross-platform anchor waves.
- [Comprehensive experiment matrix](docs/experiment_matrix.md): full source-of-truth matrix with `E001` to `E200`, priority levels, metrics, and GPU-hour estimates.
- [Local dev runbook](docs/local_dev_runbook.md): Mac/local baseline-style runs for development while HPC access is unavailable.
- [Dev experiment matrix](docs/dev_experiment_matrix.md): rationale, parameters, and local-dev versus HPC-only experiment boundaries.
- [Generated experiment tracker](results/processed/experiment_tracker.md): current source matrix status generated from configs and raw results, with separately labelled supplemental platform probes when present.
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
source slurm/modules_for_hpc_ai
DS_BUILD_OPS=0 DS_BUILD_AIO=0 python -m pip install \
  --constraint slurm/constraints/lovelace_py311_cuda124.txt \
  --upgrade-strategy only-if-needed \
  -e ".[dev,yaml,workloads,distributed]"
```

The Lovelace vLLM gate uses a private PyTorch 2.8.0/CUDA 12.4 build with Gloo and NCCL plus pinned vLLM `0.10.2`.
Do not install a generic PyPI vLLM wheel into the shared environment: its CUDA ABI may target another runtime. The L40S
and H100 E006 acceptance gates have passed, so E006 and the supported one-/two-/four-GPU GPT-2/TinyLlama vLLM rows
are scheduled in platform-separated `vllm-*` waves that activate the private environment. See the
[vLLM environment runbook](docs/vllm_environment_runbook.md) and [deferred experiment audit](docs/deferred_experiment_audit.md).

Do not use `--upgrade-strategy eager` on Lovelace because it can install PyPI Torch/Numpy over the Spack-provided CUDA
runtime stack.

Before running environment-validation waves from an allocated GPU node, check the active runtime:

```bash
python -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --expected-torch-version 2.8.0 --expected-torch-cuda 12.4 --require-deepspeed
```

The Lovelace helper resolves `CUDA_HOME` from its verified `nvcc`. Use the project `distributed` extra, which requires
DeepSpeed 0.18.9 or newer, and run `make check-cuda-toolchain PLATFORM=l40s EXPECTED_GPUS=1` from an allocated GPU node
before source builds or CUDA-op experiments.

TensorRT-LLM and FP8/quantized model tooling remain vendor/backend-specific and are not silently substituted by these
extras. vLLM waves use their separately pinned private runtime.

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

Use an immutable submission bundle for every real Slurm wave. The bundle is a detached Git worktree plus frozen
platform configs, Slurm scripts, an environment contract, and staging outside the primary checkout. This means that
importing or committing an earlier result cannot invalidate jobs that are still waiting in Slurm.

Validate the profile, then submit a complete wave with one command:

```bash
source slurm/modules_for_hpc_ai
make check-slurm-profile SLURM_PROFILE=slurm/profiles/cluster_l40s.json
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

`submit-platform-wave` creates the detached bundle, records its location under ignored `build/platform_waves/`, writes
the Slurm status file, and submits every script. No bundle path needs to be copied into the shell. The Makefile uses
`.venv/bin/python` automatically when it exists.

Monitor result coverage while the wave runs:

```bash
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
```

Each script verifies the source commit, manifest SHA-256, resolved config SHA-256, clean snapshot, and dependency
contract before running. Workload-side files, including profiler traces, are written to the bundle cache rather than
the immutable source snapshot; each trial/rank receives a separate trace path.

When the status command reports every trial complete, collect and stage the verified raw evidence:

```bash
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
git diff --cached --stat
git commit -m "data: add completed L40S wave 0 trials"
```

Collection validates every expected result and its bundle manifest, imports only complete experiments, refreshes the
checksum manifest, and updates the ignored platform tracker. If a wave has missing or failed trials, collection stops
without importing a partial dataset; run `make retry-platform-wave PLATFORM=l40s PLATFORM_WAVE=0` to submit a fresh
bundle for the incomplete experiments only.

Wave 0 uses one acceptance trial per condition. Core performance waves use three-task Slurm arrays based on the
tracked execution matrix; see the [measurement protocol](docs/measurement_protocol.md). H100 uses the same bundle
workflow with `PLATFORM=h100`; its evidence remains in a separate platform root.

The older `render-platform-wave` and `*-bundle` targets remain available for inspection and one-off recovery, but they
are not the recommended path for queued research collection.

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
uses additive counts plus the number of matched cross-platform protocols for combined KPIs; statistical averages,
confidence intervals, and scaling baselines remain platform-local. Cross-study throughput rankings and operational
submission summaries are deliberately omitted because they do not answer a controlled research question. Detailed
trial evidence remains available in a collapsed audit panel. `make check` includes mixed-platform Python and
JavaScript regression tests.
The scaling view overlays the isolated series on one GPU-count axis, assigns every series a stable colour, and uses
solid and dashed strokes for strong and weak scaling. A local selector can show all series, strong scaling only, or
weak scaling only. Its legend preserves the platform, workload, comparability, strategy, precision, baseline, and
source-condition boundaries behind every line.
The H100-versus-L40S view additionally derives campaign-wide matched pairs from exact conditions only. It currently
spans 164 protocol matches and reports unitless per-pair throughput ratios by study, workload, and execution mode;
it never averages raw throughput values across token definitions or unmatched configurations. Readers can filter the
overview and open either platform's immutable trial evidence from the selected protocol.
The energy section uses corrected measured-region telemetry only. It separates throughput units, plots throughput
against tokens per kWh, distinguishes aggregate from per-GPU power, and limits energy-to-solution comparisons to
conditions with matching platform, workload, batch, sequence length, measured steps, and GPU count.

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
