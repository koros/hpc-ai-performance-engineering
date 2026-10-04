"""Explicit workload accounting for the controlled scaling extension."""

from __future__ import annotations

from hpc_ai_perf.config.experiment import ExperimentConfig

INPUT_PROTOCOL = "synthetic_global_batches_v1"
REAL_TEXT_PROTOCOL = "wikitext103_global_batches_v1"
INPUT_PROTOCOLS = {INPUT_PROTOCOL, REAL_TEXT_PROTOCOL}
STUDY_ID = "tinyllama_batch_crossover_v1"
DEEP_DIVE_STUDY_ID = "scaling_deep_dive_v1"

# Real text uses the WikiText revision already pinned for the low-bit study.
REAL_TEXT_DATASET = "Salesforce/wikitext"
REAL_TEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
REAL_TEXT_SUBSET = "wikitext-103-raw-v1"
REAL_TEXT_BATCHING = {"pad_to_max", "dynamic_pad", "length_bucketed", "pack"}

CONTROLLED_STRATEGIES = {"ddp", "fsdp"}
FSDP_DETAILS = {"full_shard", "sharded_grad_op"}
FSDP_WRAP_POLICIES = {"root", "transformer_layer"}
DDP_COMM_HOOKS = {"none", "bf16_compress", "fp16_compress"}
DDP_KEYS = ("ddp_bucket_cap_mb", "ddp_comm_hook", "ddp_gradient_as_bucket_view", "ddp_static_graph")
TRANSFORMER_ENGINE_WORKLOADS = {"gpt2_small", "distilbert"}


def scaling_issues(config: ExperimentConfig) -> list[str]:
    p = config.workload.parameters
    if not p.get("scaling_study_id"):
        return ["input_protocol requires a scaling_study_id"] if "input_protocol" in p else []
    issues = []
    strategy = config.distributed.strategy
    if config.mode != "training" or strategy not in CONTROLLED_STRATEGIES:
        issues.append("controlled scaling currently requires training with DDP or FSDP")
    if p.get("stack") != strategy:
        issues.append("scaling stack must match distributed.strategy")
    issues.extend(_input_issues(config))
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
    issues.extend(_precision_issues(config))
    if p.get("attention_backend") != "sdpa":
        issues.append("controlled scaling requires an explicit sdpa attention backend")
    if p.get("training_update_diagnostics") is not True:
        issues.append("controlled scaling requires optimizer-update diagnostics")
    if "torch_profiler" in config.profiling.profilers:
        issues.append("profiled diagnostic runs must not use the controlled performance protocol")
    issues.extend(_stack_issues(config))
    return issues


def _input_issues(config: ExperimentConfig) -> list[str]:
    p = config.workload.parameters
    protocol = p.get("input_protocol")
    if protocol == INPUT_PROTOCOL:
        if config.workload.dataset != INPUT_PROTOCOL:
            return [f"controlled scaling requires dataset and input_protocol={INPUT_PROTOCOL}"]
        if "text_batching" in p:
            return ["text_batching applies only to the real-text input protocol"]
        return []
    if protocol == REAL_TEXT_PROTOCOL:
        issues = []
        if (config.workload.dataset, p.get("dataset_revision"), p.get("dataset_subset")) != (
            REAL_TEXT_DATASET,
            REAL_TEXT_REVISION,
            REAL_TEXT_SUBSET,
        ):
            issues.append("real-text scaling must use the pinned WikiText-103 dataset revision")
        if p.get("text_batching") not in REAL_TEXT_BATCHING:
            issues.append(f"text_batching must be one of {sorted(REAL_TEXT_BATCHING)}")
        if config.workload.name == "distilbert":
            issues.append("real-text scaling is defined for causal-LM workloads only")
        return issues
    return [f"controlled scaling requires dataset and input_protocol in {sorted(INPUT_PROTOCOLS)}"]


def _precision_issues(config: ExperimentConfig) -> list[str]:
    """Allow three explicit numeric contracts and nothing in between.

    The default keeps FP32 parameters with BF16 autocast. The precision lever adds
    pure FP32 and Transformer Engine runs (BF16 control and FP8 treatment), each
    of which must declare every dtype rather than inherit a silent default.
    """
    p = config.workload.parameters
    precision = config.runtime.precision
    load_dtype = p.get("model_load_dtype")
    if p.get("precision_backend") == "transformer_engine":
        issues = []
        if config.workload.name not in TRANSFORMER_ENGINE_WORKLOADS:
            issues.append("Transformer Engine scaling is validated for GPT-2 Small and DistilBERT only")
        if config.distributed.strategy != "ddp":
            issues.append("Transformer Engine scaling requires DDP")
        if precision not in {"bf16", "fp8"} or load_dtype != "bf16" or p.get("base_compute_dtype") != "bf16":
            issues.append("Transformer Engine scaling requires BF16 parameters with BF16 or FP8 compute")
        return issues
    if load_dtype != "fp32" or precision not in {"bf16", "fp32"}:
        return ["this scaling extension fixes FP32 parameters with BF16 or FP32 computation"]
    return []


def _stack_issues(config: ExperimentConfig) -> list[str]:
    p = config.workload.parameters
    strategy = config.distributed.strategy
    issues = []
    ddp_keys = [key for key in DDP_KEYS if key in p]
    if ddp_keys and strategy != "ddp":
        issues.append("DDP communication settings require the DDP stack")
    if "ddp_bucket_cap_mb" in p and (type(p["ddp_bucket_cap_mb"]) is not int or p["ddp_bucket_cap_mb"] < 1):
        issues.append("ddp_bucket_cap_mb must be a positive integer")
    if "ddp_comm_hook" in p and p["ddp_comm_hook"] not in DDP_COMM_HOOKS:
        issues.append(f"ddp_comm_hook must be one of {sorted(DDP_COMM_HOOKS)}")
    for key in ("ddp_gradient_as_bucket_view", "ddp_static_graph"):
        if key in p and type(p[key]) is not bool:
            issues.append(f"{key} must be a boolean")
    if p.get("ddp_static_graph") is True and p.get("gradient_accumulation_steps") != 1:
        # PyTorch's reducer asserts (expect_autograd_hooks_) when no_sync skips a static graph's sync.
        issues.append("ddp_static_graph cannot be combined with gradient accumulation")
    if strategy == "fsdp":
        if p.get("strategy_detail") not in FSDP_DETAILS:
            issues.append(f"FSDP scaling requires strategy_detail in {sorted(FSDP_DETAILS)}")
        if p.get("fsdp_wrap_policy") not in FSDP_WRAP_POLICIES:
            issues.append(f"FSDP scaling requires fsdp_wrap_policy in {sorted(FSDP_WRAP_POLICIES)}")
    elif "fsdp_wrap_policy" in p:
        issues.append("fsdp_wrap_policy applies only to the FSDP stack")
    return issues
