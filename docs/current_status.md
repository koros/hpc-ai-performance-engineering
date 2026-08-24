# Current Project Status

This snapshot summarises the repository state before full HPC submission. It should be updated after major workflow or experiment changes.

## Ready Now

| Area | Status |
| --- | --- |
| Experiment matrix | 200 concrete matrix configs are present and load successfully. |
| Validation | `python -m hpc_ai_perf.cli validate --skip-results` passes with 0 config issues. |
| Local smoke runner | CPU-only synthetic workload is ready and covered by tests. |
| Semantic launch gate | `validate --research-ready` blocks unsupported precision/backend/framework combinations before Slurm rendering or execution. |
| Result pipeline | Immutable, repetition-aware matrix trials, verified config/provenance fields, aggregation, 11 objective-specific analysis tables, progress tracking, and readiness reporting are wired into `python -m hpc_ai_perf.cli refresh`. |
| Distributed execution | One-process-per-GPU `torchrun`, DDP, FSDP, DeepSpeed ZeRO, rank binding, global work accounting, and rank-safe result writing are implemented. |
| Profiling | nvidia-smi power/utilisation/memory/clock/temperature sampling, PyTorch traces, workload timing, and NCCL microbenchmark metrics are implemented. |
| Slurm support | Semantic-gated single/bulk rendering, H100/L40S profile matching, CPU allocation checks, and three transformer smoke presets are available. |
| Dashboard | Static dashboard reads processed results, analysis summaries, and readiness report output. |
| Quality gates | `pytest`, `ruff`, and `pre-commit` are configured and passing. |

## Smoke-Ready Workloads

| Workload | Current execution support | Smoke config | Notes |
| --- | --- | --- | --- |
| `synthetic` | Ready | `configs/local/synthetic_smoke.json` | Runs without GPU or optional ML dependencies. |
| `gpt2_small` | Training plus PyTorch/vLLM inference | `configs/local/gpt2_small_smoke.json` | Matrix GPU runs require the matching HPC extras and CUDA. |
| `distilbert` | Training, PyTorch classification inference, concurrent CUDA streams, and multi-GPU data parallelism | `configs/local/distilbert_smoke.json` | Causal-generation adapters are deliberately rejected for this classifier. |
| `tinyllama` | Training plus PyTorch/vLLM inference | `configs/local/tinyllama_inference_smoke.json` | Full-size training remains optional because of GPU-hour cost. |

## Current Matrix Breakdown

| Category | Count |
| --- | ---: |
| Total matrix configs | 200 |
| Must Have | 116 |
| Should Have | 71 |
| Optional | 13 |
| Training configs | 107 |
| Inference configs | 93 |

## Before Full HPC Submission

1. Install the workload extra on the cluster:

   ```bash
   python -m pip install -e ".[dev,yaml,workloads,distributed,serving]"
   ```

2. Review Slurm smoke presets for partition, account, module loading, and virtual environment activation:

   ```bash
   slurm/smoke/gpt2_small_smoke.sbatch
   slurm/smoke/distilbert_smoke.sbatch
   slurm/smoke/tinyllama_inference_smoke.sbatch
   ```

3. Run the HPC smoke checklist:

   ```bash
   docs/hpc_smoke_runbook.md
   ```

4. Refresh generated outputs:

   ```bash
   python -m hpc_ai_perf.cli refresh
   ```

5. Review:

   - `results/processed/readiness_report.md`
   - `results/processed/experiment_tracker.md`
   - dashboard at `http://localhost:8000/dashboard/`

## Main Remaining Work

| Area | Remaining work |
| --- | --- |
| HPC verification | Run `E001-E006`, then baseline and 1/2/4-GPU pilots on Lovelace; local tests cannot validate CUDA/NCCL/Slurm behaviour. |
| Independent trials | Baseline scripts render as three-task Slurm arrays; execute them on Lovelace and inspect confidence intervals and variance from preserved trial records. |
| Backend-dependent rows | FP8, INT8, and TensorRT-LLM configs are explicitly deferred until a named backend, calibration/engine artefact, and compatible cluster environment are supplied. |
| Invalid DistilBERT framework rows | `E154-E155` must be removed from executable scope or redesigned around classification-compatible backends. |
| vLLM NCCL telemetry | `E147-E148` and `E195-E196` no longer claim request-path NCCL telemetry; they remain executable with serving-latency and system profilers. |
| Communication depth | The NCCL microbenchmark and PyTorch trace do not yet provide NVLink counters or full communication/compute attribution. |
| Quality scope | Synthetic held-out checks detect instability; final RQ3 quality claims still need a preregistered real evaluation protocol. |
| Dashboard | Trial confidence intervals and explicit insufficient-data states are connected; dedicated RQ views and conclusion-to-trial traceability remain. |

## Deferred Research Scope

Run `python -m hpc_ai_perf.cli validate --skip-results --research-ready` for the authoritative gate; it must pass with zero blockers. The readiness report lists deferred rows that are intentionally excluded from deployment batches until a TensorRT engine, selected FP8 backend, quantization artifact, or classification-compatible serving adapter is supplied. The runner still refuses to schedule those rows directly, so unsupported measurements cannot be fabricated.
