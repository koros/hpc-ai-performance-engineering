from __future__ import annotations

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.profiling import SUPPORTED_PROFILERS

SUPPORTED_WORKLOADS = {"distilbert", "environment_validation", "gpt2_small", "synthetic", "tinyllama"}
SUPPORTED_ENVIRONMENT_TARGETS = {
    "cuda_pytorch",
    "ddp_nccl_allreduce",
    "fsdp_smoke",
    "deepspeed_zero_smoke",
    "vllm_serving",
    "tensorrt_llm",
}


def capability_issues(config: ExperimentConfig) -> list[str]:
    """Return semantic problems that would invalidate the declared experiment."""
    issues: list[str] = []
    parameters = config.workload.parameters
    workload = config.workload.name
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
        is_managed_tensor_parallel = config.mode == "inference" and tensor_parallel == config.hardware.gpus
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

    if workload in {"gpt2_small", "distilbert", "tinyllama"}:
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
        if config.runtime.precision in {"fp8", "int8", "int4"}:
            issues.append(
                f"precision {config.runtime.precision} requires a backend/model-specific implementation "
                "not selected by this config"
            )
        elif config.mode == "training" and config.runtime.precision not in {"fp32", "fp16", "bf16"}:
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
            valid_tensor_parallel = selected_framework == "vllm" and tensor_parallel == config.hardware.gpus
            valid_classification_data_parallel = (
                workload == "distilbert"
                and selected_framework == "pytorch_eager"
                and data_parallel == config.hardware.gpus
            )
            if not valid_tensor_parallel and not valid_classification_data_parallel:
                issues.append(
                    "multi-GPU inference requires either vLLM tensor parallelism or DistilBERT PyTorch "
                    "data parallelism matching hardware.gpus"
                )

        input_length = int(parameters.get("prompt_length", parameters.get("sequence_length", 128)))
        generated_length = 0 if workload == "distilbert" else int(parameters.get("max_new_tokens", 32))
        context_limits = {"distilbert": 512, "gpt2_small": 1024, "tinyllama": 2048}
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
