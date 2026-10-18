from __future__ import annotations

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.config.long_context import PARAMETERS as LONG_CONTEXT_PARAMETERS
from hpc_ai_perf.config.long_context import long_context_issues
from hpc_ai_perf.config.lowbit import PARAMETERS as LOWBIT_PARAMETERS
from hpc_ai_perf.config.lowbit import lowbit_issues
from hpc_ai_perf.config.scaling import scaling_issues
from hpc_ai_perf.profiling import SUPPORTED_PROFILERS
from hpc_ai_perf.serving_layout import (
    REPLICA_LAYOUTS,
    declared_serving_layout,
    serving_layout_issues,
)

SUPPORTED_WORKLOADS = {
    "distilbert",
    "environment_validation",
    "gpt2_small",
    "gpt2_medium",
    "gpt2_large",
    "synthetic",
    "tinyllama",
    "qwen2_5_7b",
    "qwen2_5_7b_long_context",
    "qwen2_5_7b_training",
    "qwen2_5_3b",
    "qwen2_5_7b_bf16_serving",
}
SUPPORTED_ENVIRONMENT_TARGETS = {
    "cuda_pytorch",
    "ddp_nccl_allreduce",
    "fsdp_smoke",
    "deepspeed_zero_smoke",
    "vllm_serving",
    "tensorrt_llm",
}
TRANSFORMER_PARAMETER_KEYS = {
    "ignore_eos",
    "max_model_len",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "attention_backend",
    "dataset_revision",
    "dataset_subset",
    "ddp_bucket_cap_mb",
    "ddp_comm_hook",
    "ddp_gradient_as_bucket_view",
    "ddp_static_graph",
    "fsdp_wrap_policy",
    "text_batching",
    "input_protocol",
    "scaling_study_id",
    "scaling_role",
    "scaling_weak_per_gpu_batch_size",
    "activation_checkpointing",
    "allreduce_bytes",
    "allreduce_iterations",
    "base_compute_dtype",
    "baseline_kind",
    "batch_size",
    "capture_gpu_topology",
    "collect_gradient_synchronization",
    "collect_loss",
    "collect_sharded_communication",
    "comparison_control_id",
    "comparison_scope",
    "comparison_role",
    "comparison_treatment_id",
    "communication_study_type",
    "comparison_axis",
    "concurrent_requests",
    "data_movement_study_type",
    "data_parallel_size",
    "replica_count",
    "request_routing",
    "deepspeed_config",
    "deepspeed_accumulation_mode",
    "device",
    "distributed_strategy",
    "effective_batch_size",
    "engine_path",
    "fixed_workload",
    "framework",
    "framework_label",
    "frameworks",
    "fp8_amax_history_len",
    "fsdp_config",
    "global_batch_size",
    "gradient_accumulation_steps",
    "hardware_label",
    "hf_model_name",
    "hf_model_revision",
    "inference_study_type",
    "learning_rate",
    "latency_slo_p95_ms",
    "model_load_dtype",
    "max_new_tokens",
    "memory_study_type",
    "micro_batch_size",
    "min_measured_seconds",
    "num_labels",
    "num_workers",
    "per_gpu_batch_size",
    "pinned_memory",
    "precision_backend",
    "precision_label",
    "prefetch_factor",
    "profile_target",
    "prompt",
    "prompt_length",
    "quality_eval_samples",
    "quality_sanity_check",
    "quantization_bits",
    "quantization_method",
    "repeat_index",
    "requests",
    "requires_supported_fp8_backend",
    "requires_transformer_engine_or_supported_inference_backend",
    "runtime_precision_selected_for_config",
    "scaling_type",
    "sequence_length",
    "stack",
    "steps",
    "strategy_detail",
    "study_protocol",
    "tensor_parallel_size",
    "serving_layout",
    "training_update_diagnostics",
    "text",
    "warmup_steps",
}
WORKLOAD_PARAMETER_KEYS = {
    "qwen2_5_7b_long_context": LONG_CONTEXT_PARAMETERS,
    "qwen2_5_7b": LOWBIT_PARAMETERS,
    "qwen2_5_7b_training": TRANSFORMER_PARAMETER_KEYS,
    "qwen2_5_3b": TRANSFORMER_PARAMETER_KEYS,
    "qwen2_5_7b_bf16_serving": TRANSFORMER_PARAMETER_KEYS,
    "distilbert": TRANSFORMER_PARAMETER_KEYS,
    "gpt2_small": TRANSFORMER_PARAMETER_KEYS,
    "gpt2_medium": TRANSFORMER_PARAMETER_KEYS,
    "gpt2_large": TRANSFORMER_PARAMETER_KEYS,
    "tinyllama": TRANSFORMER_PARAMETER_KEYS,
    "environment_validation": {
        "allreduce_bytes",
        "capture_gpu_topology",
        "engine_api_contract",
        "engine_manifest_path",
        "engine_mode",
        "engine_path",
        "expected_checks",
        "expected_gpus",
        "expected_tensorrt_llm_version",
        "hf_model_name",
        "hf_model_revision",
        "max_batch_size",
        "max_input_tokens",
        "max_new_tokens",
        "min_generated_tokens",
        "model_path",
        "prompt",
        "prompt_token_id",
        "requests",
        "tensor_parallel_size",
        "validation_target",
        "warmup_requests",
    },
    "nanogpt": {"batch_size", "sequence_length"},
    "synthetic": {"sleep_seconds", "tokens_per_step"},
}


def unknown_parameter_issues(config: ExperimentConfig) -> list[str]:
    """Reject workload parameters that have no declared runtime or analysis meaning."""
    allowed = WORKLOAD_PARAMETER_KEYS.get(config.workload.name)
    if allowed is None:
        return []
    unknown = sorted(set(config.workload.parameters) - allowed)
    if not unknown:
        return []
    return [f"unknown workload parameter(s): {', '.join(unknown)}"]


def capability_issues(config: ExperimentConfig, *, include_deferred: bool = True) -> list[str]:
    """Return semantic problems that would invalidate the declared experiment.

    Deferred matrix rows remain visible in the planning catalogue, but are not
    part of the executable deployment scope until their external backend or
    protocol is supplied.  Normal execution and Slurm rendering keep reporting
    their capability issues; the research-readiness aggregate can exclude them
    explicitly without pretending that the row is runnable.
    """
    issues = unknown_parameter_issues(config)
    if str(config.metadata.get("execution_status", "ready")).lower() == "deferred" and not include_deferred:
        return issues
    parameters = config.workload.parameters
    issues.extend(scaling_issues(config))
    issues.extend(serving_layout_issues(config))
    workload = config.workload.name
    if workload == "qwen2_5_7b":
        issues.extend(lowbit_issues(config))
    if workload == "qwen2_5_7b_long_context":
        issues.extend(long_context_issues(config))
    strategy = config.distributed.strategy
    expected_gpus = config.distributed.nodes * config.distributed.gpus_per_node

    if workload not in SUPPORTED_WORKLOADS:
        issues.append(f"workload has no execution adapter: {workload}")
    unknown_profilers = sorted(set(config.profiling.profilers) - SUPPORTED_PROFILERS)
    if unknown_profilers:
        issues.append(f"profilers have no implementation: {', '.join(unknown_profilers)}")
    if config.profiling.enabled and not config.profiling.profilers:
        issues.append("profiling is enabled but no profilers are selected")

    if config.hardware.gpus != expected_gpus:
        tensor_parallel = int(parameters.get("tensor_parallel_size", 0))
        data_parallel = int(parameters.get("data_parallel_size", 1))
        replica_count = int(parameters.get("replica_count", 1))
        serving_layout = declared_serving_layout(parameters)
        is_managed_tensor_parallel = config.mode == "inference" and (
            tensor_parallel == config.hardware.gpus
            or (
                serving_layout in REPLICA_LAYOUTS
                and tensor_parallel * replica_count == config.hardware.gpus
                and data_parallel == replica_count
            )
        )
        if not is_managed_tensor_parallel:
            issues.append(f"hardware.gpus={config.hardware.gpus} does not match nodes*gpus_per_node={expected_gpus}")

    if config.mode == "training" and config.hardware.gpus > 1 and strategy == "none":
        issues.append("multi-GPU training requires an explicit distributed strategy")
    if (
        strategy.startswith("deepspeed")
        and "deepspeed_config" not in parameters
        and workload != "environment_validation"
    ):
        issues.append("DeepSpeed strategy is missing workload.parameters.deepspeed_config")
    if strategy == "fsdp" and config.hardware.gpus > 1 and workload != "environment_validation":
        detail = parameters.get("strategy_detail") or parameters.get("fsdp_config", {}).get("sharding_strategy")
        if detail not in {"full_shard", "sharded_grad_op"}:
            issues.append("FSDP config must select full_shard or sharded_grad_op")

    if workload == "environment_validation":
        target = str(parameters.get("validation_target", ""))
        if target not in SUPPORTED_ENVIRONMENT_TARGETS:
            issues.append(f"environment validation target has no implementation: {target}")
        if target == "vllm_serving" and (
            not parameters.get("hf_model_name") or not parameters.get("hf_model_revision")
        ):
            issues.append("vLLM environment validation requires a pinned hf_model_name and hf_model_revision")
        if target == "tensorrt_llm":
            _extend_tensorrt_llm_environment_issues(issues, parameters, config.hardware.gpus)

    if workload in {
        "gpt2_small",
        "distilbert",
        "tinyllama",
        "qwen2_5_7b_long_context",
        "qwen2_5_7b_training",
        "qwen2_5_3b",
        "qwen2_5_7b_bf16_serving",
    }:
        if config.experiment_id.startswith("E") and not config.workload.model_revision:
            issues.append("matrix transformer config must pin workload.model_revision")
        model_override = parameters.get("hf_model_name")
        if (
            config.experiment_id.startswith("E")
            and model_override
            and model_override != config.workload.model
            and not parameters.get("hf_model_revision")
        ):
            issues.append("matrix hf_model_name override must pin workload.parameters.hf_model_revision")
        # The controlled scaling contract validates its own Transformer Engine stack (DDP,
        # GPT-2/DistilBERT only); its multi-GPU runs are gated by a canary wave instead.
        controlled_scaling = bool(parameters.get("scaling_study_id"))
        if config.runtime.precision == "fp8" and config.mode == "training":
            if parameters.get("precision_backend") != "transformer_engine":
                issues.append("FP8 training requires precision_backend=transformer_engine")
            if parameters.get("quantization_method") != "fp8_delayed_scaling_hybrid":
                issues.append("FP8 training requires quantization_method=fp8_delayed_scaling_hybrid")
            if parameters.get("base_compute_dtype") != "bf16":
                issues.append("FP8 training requires base_compute_dtype=bf16")
            if not parameters.get("comparison_control_id"):
                issues.append("FP8 training requires a BF16 comparison_control_id")
            if int(parameters.get("fp8_amax_history_len", 0)) < 1:
                issues.append("FP8 training requires fp8_amax_history_len of at least 1")
            if not controlled_scaling and parameters.get("stack") != "pytorch_transformer_engine":
                issues.append("FP8 training requires stack=pytorch_transformer_engine")
            if not controlled_scaling and (config.distributed.strategy != "none" or config.hardware.gpus != 1):
                issues.append("validated Transformer Engine FP8 training currently supports one GPU")
        elif config.mode == "training" and parameters.get("precision_backend") == "transformer_engine":
            if config.runtime.precision != "bf16":
                issues.append("Transformer Engine training controls must use BF16 or the validated FP8 path")
            if parameters.get("base_compute_dtype") != "bf16":
                issues.append("Transformer Engine BF16 controls require base_compute_dtype=bf16")
            if not controlled_scaling and parameters.get("stack") != "pytorch_transformer_engine":
                issues.append("Transformer Engine BF16 controls require stack=pytorch_transformer_engine")
            if not controlled_scaling and (config.distributed.strategy != "none" or config.hardware.gpus != 1):
                issues.append("validated Transformer Engine BF16 controls currently support one GPU")
        elif config.runtime.precision == "fp8" and config.mode == "inference" and parameters.get("framework") == "vllm":
            if parameters.get("precision_backend") != "vllm_fp8_online":
                issues.append("vLLM FP8 inference requires precision_backend=vllm_fp8_online")
            if parameters.get("quantization_method") != "fp8_w8a8_dynamic":
                issues.append("vLLM FP8 inference requires quantization_method=fp8_w8a8_dynamic")
            if parameters.get("base_compute_dtype") != "bf16":
                issues.append("vLLM FP8 inference requires base_compute_dtype=bf16")
            if not parameters.get("comparison_control_id"):
                issues.append("vLLM FP8 inference requires a comparison_control_id")
        elif config.runtime.precision in {"fp8", "int8", "int4", "nf4"}:
            issues.append(
                f"precision {config.runtime.precision} requires a backend/model-specific implementation "
                "not selected by this config"
            )
        elif config.mode == "training" and config.runtime.precision not in {"fp32", "fp16", "bf16", "fp8"}:
            issues.append(f"training precision is not implemented: {config.runtime.precision}")

    framework = parameters.get("framework")
    frameworks = parameters.get("frameworks")
    if isinstance(frameworks, list) and len(frameworks) != 1:
        issues.append("frameworks must select exactly one executable backend, not a list of alternatives")
    selected_framework = str(
        framework or (frameworks[0] if isinstance(frameworks, list) and len(frameworks) == 1 else "pytorch_eager")
    )
    if selected_framework == "pytorch":
        selected_framework = "pytorch_eager"
    if config.mode == "inference":
        if "nccl" in config.profiling.profilers:
            issues.append("inference adapter does not expose request-path NCCL metrics")
        if selected_framework not in {"pytorch_eager", "vllm", "tensorrt_llm"}:
            issues.append(f"inference framework is not implemented: {selected_framework}")
        if workload == "distilbert" and selected_framework != "pytorch_eager":
            issues.append(
                "DistilBERT sequence classification is incompatible with the selected generic "
                f"{selected_framework} adapter"
            )
        if selected_framework == "tensorrt_llm" and not parameters.get("engine_path"):
            issues.append("TensorRT-LLM config requires a built engine_path")
        if (
            selected_framework == "pytorch_eager"
            and workload != "distilbert"
            and int(parameters.get("concurrent_requests", 1)) != 1
        ):
            issues.append("PyTorch eager causal-LM configs cannot claim concurrent serving requests")
        if config.hardware.gpus > 1:
            tensor_parallel = int(parameters.get("tensor_parallel_size", 0))
            data_parallel = int(parameters.get("data_parallel_size", 0))
            serving_layout = declared_serving_layout(parameters)
            valid_tensor_parallel = selected_framework == "vllm" and tensor_parallel == config.hardware.gpus
            valid_replica_layout = (
                selected_framework == "vllm"
                and serving_layout in REPLICA_LAYOUTS
                and (tensor_parallel, data_parallel) == REPLICA_LAYOUTS[serving_layout]
                and int(parameters.get("replica_count", 1)) == data_parallel
                and config.hardware.gpus == tensor_parallel * data_parallel
            )
            valid_environment_vllm_tensor_parallel = (
                workload == "environment_validation"
                and str(parameters.get("validation_target", "")) == "vllm_serving"
                and tensor_parallel == config.hardware.gpus
            )
            valid_classification_data_parallel = (
                workload == "distilbert"
                and selected_framework == "pytorch_eager"
                and data_parallel == config.hardware.gpus
            )
            if (
                not valid_tensor_parallel
                and not valid_replica_layout
                and not valid_environment_vllm_tensor_parallel
                and not valid_classification_data_parallel
            ):
                issues.append(
                    "multi-GPU inference requires either vLLM tensor parallelism or DistilBERT PyTorch "
                    "data parallelism matching hardware.gpus"
                )

        input_length = int(parameters.get("prompt_length", parameters.get("sequence_length", 128)))
        generated_length = 0 if workload == "distilbert" else int(parameters.get("max_new_tokens", 32))
        context_limits = {
            "distilbert": 512,
            "gpt2_small": 1024,
            "tinyllama": 2048,
            "qwen2_5_7b_long_context": 32768,
            "qwen2_5_3b": 32768,
            "qwen2_5_7b_bf16_serving": 32768,
        }
        context_limit = context_limits.get(workload)
        if context_limit is not None and input_length + generated_length > context_limit:
            issues.append(
                f"input plus generated length ({input_length + generated_length}) exceeds the "
                f"{workload} context window ({context_limit})"
            )

    if config.mode == "training" and workload in {"gpt2_small", "distilbert", "tinyllama"}:
        batch_size = int(
            parameters.get("micro_batch_size", parameters.get("per_gpu_batch_size", parameters.get("batch_size", 1)))
        )
        accumulation = int(parameters.get("gradient_accumulation_steps", 1))
        global_batch = batch_size * accumulation * max(1, config.hardware.gpus)
        if "effective_batch_size" in parameters and int(parameters["effective_batch_size"]) != global_batch:
            issues.append(
                f"effective_batch_size={parameters['effective_batch_size']} does not match implied "
                f"global batch {global_batch}"
            )
        if "global_batch_size" in parameters and int(parameters["global_batch_size"]) != global_batch:
            issues.append(
                f"global_batch_size={parameters['global_batch_size']} does not match implied global batch "
                f"{global_batch}"
            )

    if "data_movement_study_type" in parameters:
        workers = int(parameters.get("num_workers", -1))
        if workers < 0:
            issues.append("data-movement config must define a non-negative num_workers")
        if workers == 0 and "prefetch_factor" in parameters:
            issues.append("prefetch_factor requires num_workers greater than zero")

    return issues


def _extend_tensorrt_llm_environment_issues(
    issues: list[str], parameters: dict[str, object], hardware_gpus: int
) -> None:
    """Validate the explicit saved-engine contract used by E007."""
    required = [
        "engine_api_contract",
        "engine_mode",
        "engine_path",
        "expected_tensorrt_llm_version",
        "hf_model_name",
        "hf_model_revision",
    ]
    missing = [field for field in required if not parameters.get(field)]
    if missing:
        issues.append("TensorRT-LLM environment validation is missing: " + ", ".join(missing))
        return
    if parameters.get("engine_api_contract") != "legacy_llm_engine_v1":
        issues.append("TensorRT-LLM environment validation requires engine_api_contract=legacy_llm_engine_v1")
    engine_mode = str(parameters.get("engine_mode"))
    if engine_mode not in {"build_and_validate", "validate_prebuilt"}:
        issues.append("TensorRT-LLM engine_mode must be build_and_validate or validate_prebuilt")
    if engine_mode == "build_and_validate" and not parameters.get("model_path"):
        issues.append("TensorRT-LLM build_and_validate mode requires model_path")
    if engine_mode == "validate_prebuilt" and not parameters.get("engine_manifest_path"):
        issues.append("TensorRT-LLM validate_prebuilt mode requires engine_manifest_path")
    tensor_parallel_size = int(parameters.get("tensor_parallel_size", 1))
    if tensor_parallel_size != hardware_gpus:
        issues.append("TensorRT-LLM tensor_parallel_size must match hardware.gpus")
    for field in ["max_batch_size", "max_input_tokens", "max_new_tokens", "min_generated_tokens", "requests"]:
        if int(parameters.get(field, 1)) < 1:
            issues.append(f"TensorRT-LLM {field} must be at least 1")
    if int(parameters.get("warmup_requests", 1)) < 0:
        issues.append("TensorRT-LLM warmup_requests must be non-negative")
