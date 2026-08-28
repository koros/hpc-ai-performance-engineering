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
| Platform execution | All 200 logical rows route exactly once per platform. There are 181 planned L40S conditions and 182 planned H100 conditions, with generated configs, wave-based Slurm batches, and separate trial trackers. |
| Evidence retention | L40S/H100 JSON is imported non-destructively into separate roots, scanned for sensitive keys, validated, and covered by tracked SHA-256 manifests. |
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

## Current HPC State

The L40S node is the active execution target. The H100 node is unavailable, so H100 rows remain unrun rather than
being marked failed or replaced by L40S measurements. Follow
[the L40S-first platform runbook](platform_execution_runbook.md); the same matrix can render H100 conditions when the
node returns.

## Before Each L40S Wave

1. Install the workload extra on the cluster:

   ```bash
   python -m pip install -e ".[dev,yaml,workloads,distributed]"
   ```

   Install vLLM separately for the active CUDA/PyTorch module stack before E006 or any later vLLM row. A wheel built
   against another CUDA runtime, such as one expecting `libcudart.so.13` while the active PyTorch reports CUDA 12.x,
   must be replaced with a site-compatible vLLM install.

2. Validate the active profile and render the selected platform wave:

   ```bash
   python -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
   make render-l40s-wave PLATFORM_WAVE=0
   ```

3. Review, initialize, and dry-run the Slurm batch:

   ```bash
   make plan-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   make init-platform-wave-status PLATFORM=l40s PLATFORM_WAVE=0
   make dry-run-submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   ```

4. After completion, import and verify result evidence:

   ```bash
   python -m hpc_ai_perf.cli import-result-evidence --source build/platform_results/l40s/wave_0
   make validate-result-evidence
   make track-l40s-progress
   ```

5. Review:

   - `results/processed/readiness_report.md`
   - `results/processed/experiment_tracker.md`
   - dashboard at `http://localhost:8000/dashboard/`

## Main Remaining Work

| Area | Remaining work |
| --- | --- |
| HPC verification | Run L40S wave 0, then baseline and 1/2/4-GPU pilots. H100 wave 0 waits for node recovery; local tests cannot validate CUDA/NCCL/Slurm behaviour. |
| Independent trials | Baseline scripts render as three-task Slurm arrays; execute them on Lovelace and inspect confidence intervals and variance from preserved trial records. |
| Backend-dependent rows | FP8 and INT8 rows still require named backend/calibration artefacts. E007 now has a saved-engine build/load/request adapter and remains deferred only until target-H100 validation; TensorRT-LLM benchmark rows still require performance-adapter integration. |
| Invalid DistilBERT framework rows | `E154-E155` must be removed from executable scope or redesigned around classification-compatible backends. |
| vLLM NCCL telemetry | `E147-E148` and `E195-E196` no longer claim request-path NCCL telemetry; they remain executable with serving-latency and system profilers. |
| Communication depth | The NCCL microbenchmark and PyTorch trace do not yet provide NVLink counters or full communication/compute attribution. |
| Quality scope | Synthetic held-out checks detect instability; final RQ3 quality claims still need a preregistered real evaluation protocol. |
| Dashboard | Trial confidence intervals and explicit insufficient-data states are connected; dedicated RQ views and conclusion-to-trial traceability remain. |

## Deferred Research Scope

Run `python -m hpc_ai_perf.cli validate --skip-results --research-ready` for the authoritative gate; it must pass with zero blockers. The platform tracker lists deferred rows that are intentionally excluded from generated deployment waves. `E007@h100` is an acceptance condition for when H100 returns; `E007@l40s` stays deferred until a compatible L40S engine environment exists. Other deferred rows still require their documented FP8, quantization, or serving prerequisites.
