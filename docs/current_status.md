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
| Platform execution | All 200 logical rows route exactly once per platform. There are 145 planned L40S conditions and 144 planned H100 conditions; L40S E006 is a one-trial isolated-vLLM pilot after acceptance, while the remaining vLLM rows and TensorRT-LLM remain deferred. |
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

Both L40S and H100 are available execution targets. They remain separate evidence streams: an L40S result is never
relabelled as H100 evidence. Follow [the platform execution runbook](platform_execution_runbook.md) and create an
immutable submission bundle for each platform/wave. The bundle isolates queued work from later result imports and
commits in the primary checkout.

## Before Each L40S Wave

1. Install the workload extra on the cluster:

   ```bash
   source slurm/modules_for_hpc_ai
   DS_BUILD_OPS=0 DS_BUILD_AIO=0 python -m pip install \
     --constraint slurm/constraints/lovelace_py311_cuda124.txt \
     --upgrade-strategy only-if-needed \
     -e ".[dev,yaml,workloads,distributed]"
   ```

   Do not install a generic vLLM wheel into this environment. The platform matrix reserves E006 as a one-trial L40S
   pilot after its isolated vLLM `0.10.2` source build and acceptance gate pass; the remaining serving rows stay
   deferred. The verified Lovelace helper exposes the CUDA 12.4.1 compiler needed by that build; see the
   [vLLM environment runbook](vllm_environment_runbook.md).

2. Validate the active profile and submit the selected immutable platform wave:

   ```bash
   python -m hpc_ai_perf.cli check-slurm-profile --slurm-profile slurm/profiles/cluster_l40s.json
   python -m hpc_ai_perf.cli check-runtime --expected-gpu-model L40S --expected-torch-version 2.8.0 --expected-torch-cuda 12.4 --require-deepspeed
   make check-slurm-profile SLURM_PROFILE=slurm/profiles/cluster_l40s.json
   make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   ```

   Run `check-runtime` from an allocated GPU node. The DeepSpeed portion must pass import/accelerator checks, and
   `make check-cuda-toolchain PLATFORM=l40s EXPECTED_GPUS=1` must pass before CUDA source builds. The active wave 0
   scope is E001-E005; vLLM is not a precondition for this supported scope.

3. Monitor validated trial coverage. The bundle location and Slurm status record are saved automatically under
   ignored `build/platform_waves/`:

   ```bash
   make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   ```

4. After the status command reports all required trials complete, collect and stage verified result evidence:

   ```bash
   make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=0
   git commit -m "data: add completed L40S wave 0 trials"
   ```

5. Review:

   - `results/processed/readiness_report.md`
   - `results/processed/experiment_tracker.md`
   - dashboard at `http://localhost:8000/dashboard/`

## Main Remaining Work

| Area | Remaining work |
| --- | --- |
| HPC verification | Run the E001-E005 core wave on each platform, then baseline and 1/2/4-GPU pilots. Local tests cannot validate CUDA/NCCL/Slurm behaviour. |
| Independent trials | Baseline scripts render as three-task Slurm arrays; execute them on Lovelace and inspect confidence intervals and variance from preserved trial records. |
| Backend-dependent rows | The L40S E006 pilot follows the isolated source build and acceptance gate; remaining vLLM rows stay deferred pending its canonical evidence review. TensorRT-LLM E007 still needs a compatible engine environment and engine/model contract; FP8/INT8 rows still need named backend/calibration artefacts. |
| Invalid DistilBERT framework rows | `E154-E155` must be removed from executable scope or redesigned around classification-compatible backends. |
| vLLM NCCL telemetry | `E147-E148` and `E195-E196` remain deferred with the other vLLM rows; they do not claim unavailable request-path NCCL telemetry. |
| Communication depth | The NCCL microbenchmark and PyTorch trace do not yet provide NVLink counters or full communication/compute attribution. |
| Quality scope | Synthetic held-out checks detect instability; final RQ3 quality claims still need a preregistered real evaluation protocol. |
| Dashboard | Trial confidence intervals and explicit insufficient-data states are connected; dedicated RQ views and conclusion-to-trial traceability remain. |

## Deferred Research Scope

Run `python -m hpc_ai_perf.cli validate --skip-results --research-ready` for the authoritative gate; it must pass with zero blockers. The platform tracker lists deferred rows that are intentionally excluded from generated deployment waves. E006 is available only as the L40S `vllm-pilot` after acceptance; the remaining vLLM studies remain deferred on both platforms. E007 remains deferred pending its CUDA-toolkit/TensorRT-LLM prerequisites. Other deferred rows still require their documented FP8, quantization, or serving prerequisites.
