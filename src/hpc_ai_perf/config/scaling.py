"""Explicit workload accounting for the controlled scaling extension."""

from __future__ import annotations

from hpc_ai_perf.config.experiment import ExperimentConfig

INPUT_PROTOCOL = "synthetic_global_batches_v1"
STUDY_ID = "tinyllama_batch_crossover_v1"


def scaling_issues(config: ExperimentConfig) -> list[str]:
    p = config.workload.parameters
    if not p.get("scaling_study_id"):
        return ["input_protocol requires a scaling_study_id"] if "input_protocol" in p else []
    issues = []
    if config.mode != "training" or config.distributed.strategy != "ddp":
        issues.append("controlled scaling currently requires training with DDP")
    if p.get("stack") != config.distributed.strategy:
        issues.append("scaling stack must match distributed.strategy")
    if config.workload.dataset != INPUT_PROTOCOL or p.get("input_protocol") != INPUT_PROTOCOL:
        issues.append(f"controlled scaling requires dataset and input_protocol={INPUT_PROTOCOL}")
    kind = p.get("scaling_type")
    if kind not in {"strong", "weak"} or p.get("fixed_workload") is not (kind == "strong"):
        issues.append("fixed_workload must be true for strong scaling and false for weak scaling")
    if p.get("scaling_role") not in {"performance", "canary"}:
        issues.append("controlled scaling requires scaling_role=performance or canary")
    if any(key in p for key in ("num_workers", "data_movement_study_type", "batch_size", "micro_batch_size")):
        issues.append("controlled scaling uses per_gpu_batch_size and preloaded global inputs only")
    for key in (
        "per_gpu_batch_size",
        "global_batch_size",
        "gradient_accumulation_steps",
        "sequence_length",
        "warmup_steps",
    ):
        if type(p.get(key)) is not int or p[key] < 1:
            issues.append(f"controlled scaling requires positive integer {key}")
    if type(config.runtime.max_steps) is not int or config.runtime.max_steps < 1:
        issues.append("controlled scaling requires a positive optimizer-update count")
    if all(type(p.get(k)) is int for k in ("per_gpu_batch_size", "global_batch_size", "gradient_accumulation_steps")):
        observed = config.hardware.gpus * p["per_gpu_batch_size"] * p["gradient_accumulation_steps"]
        if observed != p["global_batch_size"]:
            issues.append("controlled scaling global batch must equal GPUs * micro-batch * accumulation")
    if p.get("model_load_dtype") != "fp32" or config.runtime.precision != "bf16":
        issues.append("this scaling extension fixes FP32 parameters and BF16 computation")
    if p.get("attention_backend") != "sdpa":
        issues.append("controlled scaling requires an explicit sdpa attention backend")
    if p.get("training_update_diagnostics") is not True:
        issues.append("controlled scaling requires optimizer-update diagnostics")
    if "torch_profiler" in config.profiling.profilers:
        issues.append("profiled diagnostic runs must not use the controlled performance protocol")
    return issues
