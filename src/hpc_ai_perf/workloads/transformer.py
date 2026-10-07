from __future__ import annotations

import importlib.util
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.config.scaling import INPUT_PROTOCOLS, REAL_TEXT_PROTOCOL, scaling_issues
from hpc_ai_perf.profiling import measured_region
from hpc_ai_perf.results.contract import UnsupportedWorkloadError
from hpc_ai_perf.runtime_checks import disable_deepspeed_nvtx_annotations, prepare_deepspeed_import_environment
from hpc_ai_perf.serving_layout import REPLICA_LAYOUT, declared_serving_layout
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.scaling import (
    GlobalTrainingInputs,
    capture_training_placement,
    load_real_text_inputs,
    scaling_runtime_sha256,
)


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool


@dataclass(frozen=True)
class TransformerWorkload(Workload):
    """Base adapter for research transformer workloads."""

    name: str
    display_name: str
    required_packages: tuple[str, ...] = ("torch", "transformers")

    def prepare(self, config: ExperimentConfig) -> None:
        required = list(self.required_packages)
        if config.mode == "training" and _uses_transformer_engine(config):
            required.append("transformer_engine")
        if config.distributed.strategy.startswith("deepspeed"):
            required.append("deepspeed")
        if config.mode == "inference" and _selected_framework(config) == "vllm":
            required.append("vllm")
        if self.name == "distilbert" and bool(config.workload.parameters.get("quality_sanity_check", False)):
            required.append("datasets")
        missing = [package for package in required if importlib.util.find_spec(package) is None]
        if missing:
            package_list = ", ".join(missing)
            raise RuntimeError(
                f"{self.display_name} requires optional dependencies: {package_list}. "
                "Install the core workload/distributed extras with "
                "python -m pip install -e '.[workloads,distributed]'. "
                "If this batch needs vLLM, install a vLLM wheel or source build that matches the cluster CUDA/PyTorch "
                "runtime before rerunning."
            )

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        raise NotImplementedError(f"{self.display_name} does not support mode: {config.mode}")


class GPT2SmallWorkload(TransformerWorkload):
    def __init__(self) -> None:
        super().__init__(name="gpt2_small", display_name="GPT-2 Small")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode in {"training", "profiling"}:
            return _run_causal_training(config)
        if config.mode == "inference":
            return _run_causal_lm_inference(config)
        return super().run(config)


class GPT2MediumWorkload(GPT2SmallWorkload):
    """GPT-2 Medium (355M): the middle rung of the scaling study's model-size ladder."""

    def __init__(self) -> None:
        TransformerWorkload.__init__(self, name="gpt2_medium", display_name="GPT-2 Medium")


class GPT2LargeWorkload(GPT2SmallWorkload):
    """GPT-2 Large (774M): same architecture and tokenizer as GPT-2 Small, six times the parameters."""

    def __init__(self) -> None:
        TransformerWorkload.__init__(self, name="gpt2_large", display_name="GPT-2 Large")


class DistilBERTWorkload(TransformerWorkload):
    def __init__(self) -> None:
        super().__init__(name="distilbert", display_name="DistilBERT")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode in {"training", "profiling"}:
            return _run_classification_training(config)
        if config.mode == "inference":
            return _run_classification_inference(config)
        return super().run(config)


class TinyLlamaWorkload(TransformerWorkload):
    def __init__(self) -> None:
        super().__init__(name="tinyllama", display_name="TinyLlama")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode in {"training", "profiling"}:
            return _run_causal_training(config)
        if config.mode == "inference":
            return _run_causal_inference(config)
        return super().run(config)


class Qwen25SevenBTrainingWorkload(TransformerWorkload):
    """Causal-LM training adapter for the H100-only Qwen sharding study.

    This is deliberately distinct from the vLLM serving adapters for the same
    model family.  The matrix uses it only for controlled BF16 FSDP/ZeRO-3
    training comparisons and never pools those rows with the smaller-model
    strategy studies.
    """

    def __init__(self) -> None:
        super().__init__(name="qwen2_5_7b_training", display_name="Qwen2.5-7B training")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode in {"training", "profiling"}:
            return _run_causal_training(config)
        return super().run(config)


class Qwen25ThreeBWorkload(TransformerWorkload):
    """Causal-LM adapter for Qwen2.5-3B-Instruct.

    Same family/tokenizer as the 7B model used elsewhere in the matrix, so
    comparing this workload against TinyLlama (1.1B) and Qwen2.5-7B isolates
    model size as the sole variable in strategy/scaling/serving studies. Uses
    the generic causal-LM training and inference paths -- identical treatment
    to GPT-2 Small, DistilBERT, and TinyLlama above, never the specialised
    low-bit serving adapter.
    """

    def __init__(self) -> None:
        super().__init__(name="qwen2_5_3b", display_name="Qwen2.5-3B")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode in {"training", "profiling"}:
            return _run_causal_training(config)
        if config.mode == "inference":
            return _run_causal_inference(config)
        return super().run(config)


class Qwen25SevenBServingWorkload(TransformerWorkload):
    """Generic BF16 vLLM serving adapter for Qwen2.5-7B-Instruct.

    Deliberately separate from both ``Qwen25SevenBTrainingWorkload`` (training
    only) and the ``qwen2_5_7b`` low-bit serving workload in
    ``hpc_ai_perf.workloads.lowbit`` (a pinned, versioned, single-GPU-only
    quantization contract -- see ``config/lowbit.py``'s ``lowbit_issues``).
    This adapter exists to test tensor-parallel BF16 serving (TP=2/4), which
    the low-bit contract does not support and is not being relaxed to support
    here. It reuses the same generic vLLM path already proven for GPT-2
    Small, DistilBERT, and TinyLlama multi-GPU inference (``tensor_parallel_size``
    read from ``workload.parameters``), so it carries no new execution risk.
    """

    def __init__(self) -> None:
        super().__init__(name="qwen2_5_7b_bf16_serving", display_name="Qwen2.5-7B BF16 serving")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode == "inference":
            return _run_causal_inference(config)
        return super().run(config)


def transformer_workloads() -> list[TransformerWorkload]:
    return [
        GPT2SmallWorkload(),
        GPT2MediumWorkload(),
        GPT2LargeWorkload(),
        DistilBERTWorkload(),
        TinyLlamaWorkload(),
        Qwen25SevenBTrainingWorkload(),
        Qwen25ThreeBWorkload(),
        Qwen25SevenBServingWorkload(),
    ]


def _run_causal_training(config: ExperimentConfig) -> WorkloadRunOutput:
    torch, auto_model = _load_causal_lm_stack()
    runtime = _prepare_runtime(torch, config)
    parameters = config.workload.parameters
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    model_dtype = _training_model_dtype(torch, config)
    attention_kwargs = (
        {"attn_implementation": parameters["attention_backend"]} if "attention_backend" in parameters else {}
    )
    model = auto_model.from_pretrained(
        model_name,
        revision=_selected_model_revision(config, model_name),
        torch_dtype=model_dtype,
        **attention_kwargs,
    )
    return _run_training_loop(torch, config, runtime, model, task="causal_lm", model_name=model_name)


def _run_classification_training(config: ExperimentConfig) -> WorkloadRunOutput:
    torch, auto_model = _load_sequence_classification_stack()
    runtime = _prepare_runtime(torch, config)
    parameters = config.workload.parameters
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    num_labels = int(parameters.get("num_labels", 2))
    model_dtype = _training_model_dtype(torch, config)
    attention_kwargs = (
        {"attn_implementation": parameters["attention_backend"]} if "attention_backend" in parameters else {}
    )
    model = auto_model.from_pretrained(
        model_name,
        revision=_selected_model_revision(config, model_name),
        num_labels=num_labels,
        torch_dtype=model_dtype,
        ignore_mismatched_sizes=True,
        **attention_kwargs,
    )
    return _run_training_loop(
        torch,
        config,
        runtime,
        model,
        task="classification",
        model_name=model_name,
        num_labels=num_labels,
    )


def _training_model_dtype(torch: Any, config: ExperimentConfig) -> Any:
    """Return the checkpoint loading dtype without changing legacy defaults.

    Earlier experiments loaded ordinary training checkpoints in FP32 unless
    they used Transformer Engine.  Large-model BF16 sharding trials must load
    BF16 parameters before wrapping, so they opt in through an explicit,
    validated parameter instead of silently changing historical conditions.
    """

    requested = str(config.workload.parameters.get("model_load_dtype", "")).lower()
    if requested in {"", "fp32"}:
        return torch.bfloat16 if _uses_transformer_engine(config) else torch.float32
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    raise ValueError("model_load_dtype must be one of fp32, bf16, or fp16")


def _run_training_loop(
    torch: Any,
    config: ExperimentConfig,
    runtime: DistributedRuntime,
    model: Any,
    *,
    task: str,
    model_name: str,
    num_labels: int = 2,
) -> WorkloadRunOutput:
    parameters = config.workload.parameters
    steps = int(config.runtime.max_steps or parameters.get("steps", 1))
    input_protocol = parameters.get("input_protocol")
    controlled_inputs = input_protocol in INPUT_PROTOCOLS
    if issues := scaling_issues(config):
        raise ValueError("Invalid scaling contract: " + "; ".join(issues))
    if controlled_inputs:
        model.config.use_cache = False
    warmup_steps = _warmup_steps(parameters, steps)
    batch_size = _per_gpu_batch_size(parameters)
    sequence_length = int(parameters.get("sequence_length", 128))
    accumulation_steps = int(parameters.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    _validate_batch_semantics(config, runtime, batch_size, accumulation_steps)
    learning_rate = float(parameters.get("learning_rate", 5e-5))
    update_diagnostics = bool(parameters.get("training_update_diagnostics", False))
    accumulation_mode = str(parameters.get("deepspeed_accumulation_mode", "legacy_external"))
    if accumulation_mode not in {"legacy_external", "explicit_boundary_v1"}:
        raise ValueError("Unknown deepspeed_accumulation_mode")
    device = torch.device("cuda", runtime.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    activation_checkpointing_backend = "disabled"
    if bool(parameters.get("activation_checkpointing", False)):
        activation_checkpointing_backend = _enable_activation_checkpointing(
            model,
            prefer_non_reentrant=config.distributed.strategy == "fsdp",
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    fp8_context = None
    transformer_engine_metadata: dict[str, int | float | str | bool | None] = {}
    if _uses_transformer_engine(config):
        model, fp8_context, transformer_engine_metadata = _configure_transformer_engine_training(
            torch,
            config,
            model,
            device,
        )

    model, optimizer, engine, deepspeed_nvtx_annotations_disabled, strategy_runtime_metadata = (
        _configure_training_strategy(
            torch,
            config,
            runtime,
            model,
            device,
            learning_rate,
            batch_size,
            accumulation_steps,
        )
    )
    deepspeed_managed_precision = engine is not None
    amp_context, scaler = _training_precision(
        torch,
        config.runtime.precision,
        device,
        managed_by_deepspeed=deepspeed_managed_precision,
        fp8_context=fp8_context,
    )
    data_iterator = _training_data_iterator(
        torch,
        config,
        runtime,
        task=task,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=int(_model_config(model).vocab_size),
        num_labels=num_labels,
        required_batches=(warmup_steps + steps) * accumulation_steps,
    )
    static_batch = None
    global_inputs = None
    if input_protocol == REAL_TEXT_PROTOCOL:
        global_inputs = load_real_text_inputs(
            torch,
            config,
            runtime,
            device,
            model_name,
            _selected_model_revision(config, model_name),
            measured_updates=steps,
        )
    elif controlled_inputs:
        global_inputs = GlobalTrainingInputs(
            torch,
            config,
            runtime,
            device,
            int(_model_config(model).vocab_size),
            task=task,
            num_labels=num_labels,
        )
    elif data_iterator is None:
        static_batch = _make_device_batch(
            torch,
            task,
            batch_size,
            sequence_length,
            int(_model_config(model).vocab_size),
            num_labels,
            device,
            config.runtime.seed + runtime.rank,
        )

    placement_metrics = {}
    if parameters.get("capture_gpu_topology"):
        placement_metrics = capture_training_placement(torch, runtime)
    trace_enabled = config.profiling.enabled and "torch_profiler" in config.profiling.profilers

    def trace(name):
        return torch.profiler.record_function(name) if trace_enabled else nullcontext()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    data_loading_seconds = 0.0
    transfer_seconds = 0.0
    final_loss = 0.0
    initial_loss: float | None = None
    optimizer_step_calls = 0
    backward_calls = 0
    engine_steps_before = getattr(engine, "global_steps", None)
    engine_samples_before = getattr(engine, "global_samples", None)

    def one_optimizer_step(measured: bool) -> float:
        nonlocal data_loading_seconds, transfer_seconds
        nonlocal initial_loss, optimizer_step_calls, backward_calls
        if engine is not None:
            engine.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_load = 0.0
        step_transfer = 0.0
        for accumulation_index in range(accumulation_steps):
            if engine is not None and accumulation_mode == "explicit_boundary_v1":
                engine.set_gradient_accumulation_boundary(accumulation_index == accumulation_steps - 1)
            if global_inputs is not None:
                batch = global_inputs.next_batch()
            elif data_iterator is None:
                batch = static_batch
            else:
                load_start = time.perf_counter()
                try:
                    batch = next(data_iterator)
                except StopIteration as exc:
                    raise RuntimeError("DataLoader exhausted before measurement completed") from exc
                step_load += time.perf_counter() - load_start
                transfer_start = time.perf_counter()
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                _sync(torch, device)
                step_transfer += time.perf_counter() - transfer_start

            sync_context = nullcontext()
            if accumulation_index < accumulation_steps - 1 and hasattr(model, "no_sync") and engine is None:
                sync_context = model.no_sync()
            with sync_context:
                with trace("training/forward"), amp_context():
                    outputs = model(**batch)
                    if update_diagnostics and initial_loss is None:
                        initial_loss = float(outputs.loss.detach().cpu())
                    loss = outputs.loss / accumulation_steps
                with trace("training/backward_and_gradient_sync"):
                    if engine is not None:
                        if accumulation_mode == "explicit_boundary_v1":
                            # The shared loop already divided the loss by GAS.
                            engine.backward(loss, scale_wrt_gas=False)
                        else:
                            engine.backward(loss)
                    elif scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                step_loss += float(loss.detach().cpu())
                backward_calls += 1

        with trace("training/optimizer_update"):
            if engine is not None:
                engine.step()
            elif scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        optimizer_step_calls += 1
        if measured:
            data_loading_seconds += step_load
            transfer_seconds += step_transfer
        return step_loss

    model.train()
    for _ in range(warmup_steps):
        one_optimizer_step(False)
    _sync(torch, device)

    step_times: list[float] = []
    with measured_region():
        cuda_timer = _start_cuda_timer(torch)
        measured_start = time.perf_counter()
        for _ in range(steps):
            step_start = time.perf_counter()
            final_loss = one_optimizer_step(True)
            _sync(torch, device)
            step_times.append(time.perf_counter() - step_start)
        cuda_event_seconds = _stop_cuda_timer(torch, cuda_timer)
        _sync(torch, device)
        measured_seconds = max(time.perf_counter() - measured_start, 1e-12)
    measured_seconds = _distributed_max(torch, measured_seconds, device, runtime)

    total_samples = steps * accumulation_steps * batch_size * runtime.world_size
    total_tokens = total_samples * sequence_length
    real_text_metrics: dict[str, int | float | str | bool | None] = {}
    if input_protocol == REAL_TEXT_PROTOCOL:
        # Throughput counts real (non-padding) tokens so batching modes compare fairly.
        real_text_metrics = {
            "nominal_tokens": total_tokens,
            "processed_tokens": global_inputs.processed_tokens,
            "padding_fraction": 1 - global_inputs.useful_tokens / global_inputs.processed_tokens,
            "rank_token_imbalance": global_inputs.imbalance,
            "truncated_documents": global_inputs.truncated_documents,
            "text_batching": global_inputs.mode,
            "processed_tokens_sec": global_inputs.processed_tokens / measured_seconds,
        }
        total_tokens = global_inputs.useful_tokens
    final_loss = _distributed_mean(torch, final_loss, device, runtime)
    metrics: dict[str, int | float | str | bool | None] = {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "batch_size_per_gpu": batch_size,
        "effective_global_batch_size": batch_size * accumulation_steps * runtime.world_size,
        "gradient_accumulation_steps": accumulation_steps,
        "sequence_length": sequence_length,
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "throughput_unit": "training_tokens_per_second",
        "loss": final_loss,
        "measured_seconds": measured_seconds,
        "gpu_seconds": measured_seconds * runtime.world_size,
        "gpu_hours": measured_seconds * runtime.world_size / 3600,
        "throughput_samples_sec": total_samples / measured_seconds,
        "throughput_tokens_sec": total_tokens / measured_seconds,
        "step_time_mean_ms": statistics.fmean(step_times) * 1000,
        "step_time_p50_ms": _percentile(step_times, 50) * 1000,
        "step_time_p95_ms": _percentile(step_times, 95) * 1000,
        "data_loading_seconds": data_loading_seconds,
        "cpu_to_gpu_transfer_seconds": transfer_seconds,
        "device": str(device),
        "hf_model_name": model_name,
        "hf_model_revision": _selected_model_revision(config, model_name),
        "model_load_dtype_requested": str(
            parameters.get("model_load_dtype", "bf16" if _uses_transformer_engine(config) else "fp32")
        ),
        "strategy_observed": config.distributed.strategy,
        "strategy_detail_observed": _strategy_detail(config),
        "world_size": runtime.world_size,
        "rank": runtime.rank,
        "activation_checkpointing": bool(parameters.get("activation_checkpointing", False)),
        "activation_checkpointing_backend": activation_checkpointing_backend,
        "precision_backend": _precision_backend(
            config.runtime.precision,
            managed_by_deepspeed=deepspeed_managed_precision,
        ),
        "compute_dtype_observed": "fp8_mixed_bf16" if fp8_context is not None else config.runtime.precision,
        "parameter_dtype_observed": str(next(model.parameters()).dtype),
        "dataset_execution": "cpu_dataloader" if data_iterator is not None else "synthetic_device_tensors",
        "deepspeed_nvtx_annotations_disabled": deepspeed_nvtx_annotations_disabled,
        **strategy_runtime_metadata,
        **transformer_engine_metadata,
        **placement_metrics,
    }
    if global_inputs is not None:
        if global_inputs.index != global_inputs.batches:
            raise RuntimeError("Controlled input consumption differs from declared optimizer updates")
        metrics.update(
            {
                **real_text_metrics,
                "dataset_execution": input_protocol,
                "global_input_sha256": global_inputs.sha256,
                "input_micro_batches_consumed": global_inputs.index,
                "scaling_runtime_sha256": scaling_runtime_sha256(),
                "attention_backend_observed": _model_config(model)._attn_implementation,
                "training_use_cache": _model_config(model).use_cache,
            }
        )
    if cuda_event_seconds is not None:
        metrics["cuda_event_seconds"] = cuda_event_seconds
    if update_diagnostics:
        from hpc_ai_perf.validation.qwen7b_training import update_runtime_sha256

        metrics.update(
            {
                "training_update_protocol": "explicit_accumulation_audit_v1",
                "training_update_runtime_sha256": update_runtime_sha256(),
                "initial_training_loss": _distributed_mean(torch, float(initial_loss), device, runtime),
                "optimizer_step_calls": optimizer_step_calls,
                "backward_calls": backward_calls,
                "expected_optimizer_step_calls": warmup_steps + steps,
                "expected_backward_calls": (warmup_steps + steps) * accumulation_steps,
                "optimizer_state_dtypes_observed": _optimizer_state_dtypes(optimizer),
                "deepspeed_accumulation_mode": accumulation_mode if engine is not None else "common_loop",
                "deepspeed_optimizer_updates": (
                    engine.global_steps - engine_steps_before if engine_steps_before is not None else None
                ),
                "deepspeed_samples_accounted": (
                    engine.global_samples - engine_samples_before if engine_samples_before is not None else None
                ),
            }
        )
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
        metrics["memory_used_gb"] = _distributed_max(torch, allocated_gb, device, runtime)
        metrics["memory_reserved_gb"] = _distributed_max(torch, reserved_gb, device, runtime)
    if "nccl" in config.profiling.profilers and runtime.world_size > 1:
        metrics.update(_benchmark_allreduce(torch, runtime, device, parameters))
    if bool(parameters.get("quality_sanity_check", False)):
        quality_batch = _make_device_batch(
            torch,
            task,
            batch_size,
            sequence_length,
            int(_model_config(model).vocab_size),
            num_labels,
            device,
            config.runtime.seed + 1_000_000,
        )
        model.eval()
        with torch.inference_mode(), amp_context():
            quality_output = model(**quality_batch)
        quality_loss = _distributed_mean(torch, float(quality_output.loss.detach().cpu()), device, runtime)
        metrics["quality_loss"] = quality_loss
        metrics["quality_finite"] = math.isfinite(quality_loss)
        metrics["quality_protocol"] = "fixed_heldout_synthetic_v1"
        if task == "causal_lm":
            metrics["quality_perplexity"] = math.exp(min(20.0, quality_loss))
        else:
            correct = (quality_output.logits.argmax(dim=-1) == quality_batch["labels"]).float().mean()
            metrics["quality_synthetic_accuracy"] = _distributed_mean(
                torch,
                float(correct.detach().cpu()),
                device,
                runtime,
            )
    return WorkloadRunOutput(metrics=metrics)


def _optimizer_state_dtypes(optimizer: Any) -> str:
    """Observe actual optimizer state without assuming BF16 implies BF16 Adam states."""
    visited: set[int] = set()
    dtypes: set[str] = set()
    while optimizer is not None and id(optimizer) not in visited:
        visited.add(id(optimizer))
        for state in getattr(optimizer, "state", {}).values():
            for name, value in state.items():
                if name != "step" and hasattr(value, "dtype"):
                    dtypes.add(str(value.dtype))
        optimizer = getattr(optimizer, "optimizer", None)
    return ",".join(sorted(dtypes)) or "unavailable"


def _run_causal_inference(config: ExperimentConfig) -> WorkloadRunOutput:
    framework = _selected_framework(config)
    if framework == "vllm":
        return _run_vllm_inference(config)
    if framework == "tensorrt_llm":
        raise RuntimeError(
            "TensorRT-LLM requires a built engine and engine-specific benchmark; "
            "it cannot use the generic Hugging Face adapter"
        )
    if framework != "pytorch_eager":
        raise ValueError(f"Unsupported causal inference framework: {framework}")

    torch, auto_model = _load_causal_lm_stack()
    runtime = _prepare_runtime(torch, config, allow_distributed_strategy_none=True)
    if runtime.world_size != 1:
        raise RuntimeError("PyTorch eager inference is single-process; use vLLM for multi-GPU inference")
    parameters = config.workload.parameters
    device = torch.device("cuda", runtime.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = _inference_dtype(torch, config.runtime.precision, device)
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    model = auto_model.from_pretrained(
        model_name,
        revision=_selected_model_revision(config, model_name),
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    batch_size = int(parameters.get("batch_size", 1))
    concurrency = int(parameters.get("concurrent_requests", 1))
    if concurrency != 1:
        raise RuntimeError("PyTorch eager adapter cannot claim serving concurrency greater than one")
    prompt_length = int(parameters.get("prompt_length", 128))
    max_new_tokens = int(parameters.get("max_new_tokens", 32))
    _validate_context_window(config.workload.name, prompt_length, max_new_tokens)
    measured_batches = int(config.runtime.max_steps or parameters.get("requests", 10))
    warmup_batches = _warmup_steps(parameters, measured_batches)
    generator = torch.Generator(device=device).manual_seed(config.runtime.seed)
    inputs = torch.randint(
        0,
        int(model.config.vocab_size),
        (batch_size, prompt_length),
        generator=generator,
        device=device,
    )
    attention_mask = torch.ones_like(inputs)
    generate_kwargs = {
        "input_ids": inputs,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": getattr(model.config, "eos_token_id", 0) or 0,
    }
    for _ in range(warmup_batches):
        with torch.inference_mode():
            model.generate(**generate_kwargs)
    _sync(torch, device)

    latencies: list[float] = []
    generated_tokens = 0
    with measured_region():
        cuda_timer = _start_cuda_timer(torch)
        measured_start = time.perf_counter()
        for _ in range(measured_batches):
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(**generate_kwargs)
            _sync(torch, device)
            latencies.append(time.perf_counter() - start)
            generated_tokens += int(outputs.shape[-1] - prompt_length) * batch_size
        cuda_event_seconds = _stop_cuda_timer(torch, cuda_timer)
        _sync(torch, device)
        measured_seconds = max(time.perf_counter() - measured_start, 1e-12)

    ttft_samples: list[float] = []
    for _ in range(min(measured_batches, 20)):
        start = time.perf_counter()
        with torch.inference_mode():
            model.generate(**{**generate_kwargs, "max_new_tokens": 1})
        _sync(torch, device)
        ttft_samples.append(time.perf_counter() - start)

    metrics: dict[str, int | float | str | bool | None] = {
        "requests": measured_batches * batch_size,
        "batches": measured_batches,
        "warmup_batches": warmup_batches,
        "batch_size": batch_size,
        "concurrent_requests": concurrency,
        "prompt_tokens_per_request": prompt_length,
        "generated_tokens": generated_tokens,
        "total_tokens": generated_tokens,
        "throughput_unit": "generated_tokens_per_second",
        "measured_seconds": measured_seconds,
        "gpu_seconds": measured_seconds,
        "gpu_hours": measured_seconds / 3600,
        "throughput_tokens_sec": generated_tokens / measured_seconds,
        "throughput_requests_sec": measured_batches * batch_size / measured_seconds,
        "latency_p50_ms": _percentile(latencies, 50) * 1000,
        "latency_p95_ms": _percentile(latencies, 95) * 1000,
        "time_to_first_token_p50_ms": _percentile(ttft_samples, 50) * 1000,
        "time_to_first_token_p95_ms": _percentile(ttft_samples, 95) * 1000,
        "device": str(device),
        "hf_model_name": model_name,
        "hf_model_revision": _selected_model_revision(config, model_name),
        "framework_observed": "pytorch_eager",
        "precision_backend": _precision_backend(config.runtime.precision),
        "compute_dtype_observed": str(dtype),
        "parameter_dtype_observed": str(next(model.parameters()).dtype),
    }
    metrics.update(
        _serving_tail_metrics(
            latencies,
            ttft_samples,
            [],
            measured_seconds,
            parameters.get("latency_slo_p95_ms"),
        )
    )
    if cuda_event_seconds is not None:
        metrics["cuda_event_seconds"] = cuda_event_seconds
    if bool(parameters.get("quality_sanity_check", False)):
        with torch.inference_mode():
            quality_loss = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs).loss
        quality_value = float(quality_loss.detach().cpu())
        metrics["quality_loss"] = quality_value
        metrics["quality_perplexity"] = math.exp(min(20.0, quality_value))
    if torch.cuda.is_available():
        metrics["memory_used_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
    return WorkloadRunOutput(
        metrics=metrics,
        samples={"latency_seconds": latencies, "time_to_first_token_seconds": ttft_samples},
    )


def _run_classification_inference(config: ExperimentConfig) -> WorkloadRunOutput:
    framework = _selected_framework(config)
    if framework != "pytorch_eager":
        raise RuntimeError(
            f"DistilBERT sequence classification is not executed by {framework}; "
            "use pytorch_eager or define a validated classification backend"
        )
    torch, auto_model = _load_sequence_classification_stack()
    runtime = _prepare_runtime(torch, config, allow_distributed_strategy_none=True)
    parameters = config.workload.parameters
    device = torch.device("cuda", runtime.local_rank) if torch.cuda.is_available() else torch.device("cpu")
    dtype = _inference_dtype(torch, config.runtime.precision, device)
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    num_labels = int(parameters.get("num_labels", 2))
    model = auto_model.from_pretrained(
        model_name,
        revision=_selected_model_revision(config, model_name),
        num_labels=num_labels,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()
    vocab_size = int(model.config.vocab_size)
    batch_size = int(parameters.get("batch_size", 1))
    concurrency = int(parameters.get("concurrent_requests", 1))
    if batch_size < 1 or concurrency < 1:
        raise ValueError("batch_size and concurrent_requests must both be at least 1")
    sequence_length = int(parameters.get("prompt_length", parameters.get("sequence_length", 128)))
    _validate_context_window(config.workload.name, sequence_length, 0)
    data_parallel_size = int(parameters.get("data_parallel_size", 1))
    if config.hardware.gpus > 1:
        if data_parallel_size != config.hardware.gpus:
            raise RuntimeError(
                f"data_parallel_size={data_parallel_size} does not match requested GPUs={config.hardware.gpus}"
            )
        if concurrency != 1:
            raise RuntimeError("The validated multi-GPU classification path requires concurrent_requests=1")
        model = torch.nn.DataParallel(model, device_ids=list(range(data_parallel_size)))
    elif data_parallel_size != 1:
        raise RuntimeError("data_parallel_size greater than one requires a multi-GPU allocation")
    measured_batches = int(config.runtime.max_steps or 10)
    minimum_seconds = float(parameters.get("min_measured_seconds", 0))
    if not math.isfinite(minimum_seconds) or minimum_seconds < 0:
        raise ValueError("min_measured_seconds must be finite and non-negative")
    warmup_batches = _warmup_steps(parameters, measured_batches)
    generator = torch.Generator(device=device).manual_seed(config.runtime.seed)
    inputs = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=device,
    )
    attention_mask = torch.ones_like(inputs)
    for _ in range(warmup_batches):
        _classification_round(torch, model, inputs, attention_mask, device, concurrency)
    _sync(torch, device)
    latencies: list[float] = []
    logits = None
    completed_batches = 0
    with measured_region():
        cuda_timer = _start_cuda_timer(torch)
        start_all = time.perf_counter()
        for _ in range(measured_batches):
            round_logits, round_latencies = _classification_round(
                torch,
                model,
                inputs,
                attention_mask,
                device,
                concurrency,
            )
            logits = round_logits[-1]
            latencies.extend(round_latencies)
            completed_batches += 1
            if minimum_seconds and completed_batches >= 2 and time.perf_counter() - start_all >= minimum_seconds:
                break
        cuda_event_seconds = _stop_cuda_timer(torch, cuda_timer)
        _sync(torch, device)
        measured_seconds = max(time.perf_counter() - start_all, 1e-12)
    if measured_seconds < minimum_seconds:
        raise RuntimeError("Measured interval too short; increase max_steps for the energy anchor")
    total_samples = completed_batches * batch_size * concurrency
    total_tokens = total_samples * sequence_length
    metrics: dict[str, int | float | str | bool | None] = {
        "requests": total_samples,
        "samples": total_samples,
        "batches": completed_batches,
        "warmup_batches": warmup_batches,
        "batch_size": batch_size,
        "concurrent_requests": concurrency,
        "input_tokens": total_tokens,
        "total_tokens": total_tokens,
        "throughput_unit": "input_tokens_per_second",
        "predicted_label": int(logits.argmax(dim=-1).detach().cpu()[0]) if logits is not None else None,
        "measured_seconds": measured_seconds,
        "gpu_seconds": measured_seconds * data_parallel_size,
        "gpu_hours": measured_seconds * data_parallel_size / 3600,
        "throughput_requests_sec": total_samples / measured_seconds,
        "throughput_samples_sec": total_samples / measured_seconds,
        "throughput_tokens_sec": total_tokens / measured_seconds,
        "latency_p50_ms": _percentile(latencies, 50) * 1000,
        "latency_p95_ms": _percentile(latencies, 95) * 1000,
        "device": str(device),
        "hf_model_name": model_name,
        "hf_model_revision": _selected_model_revision(config, model_name),
        "framework_observed": "pytorch_eager",
        "data_parallel_size_observed": data_parallel_size,
        "precision_backend": _precision_backend(config.runtime.precision),
        "compute_dtype_observed": str(dtype),
        "parameter_dtype_observed": str(next(model.parameters()).dtype),
    }
    if cuda_event_seconds is not None:
        metrics["cuda_event_seconds"] = cuda_event_seconds
    if torch.cuda.is_available():
        metrics["memory_used_gb"] = max(
            torch.cuda.max_memory_allocated(index) / (1024**3) for index in range(data_parallel_size)
        )
    if bool(parameters.get("quality_sanity_check", False)):
        metrics.update(
            _classification_quality_metrics(
                torch,
                model,
                model_name,
                _selected_model_revision(config, model_name),
                device,
                sequence_length,
                int(parameters.get("quality_eval_samples", 64)),
            )
        )
    return WorkloadRunOutput(metrics=metrics, samples={"latency_seconds": latencies})


def _classification_round(
    torch: Any,
    model: Any,
    inputs: Any,
    attention_mask: Any,
    device: Any,
    concurrency: int,
) -> tuple[list[Any], list[float]]:
    """Execute one simultaneous classification-request group."""
    if concurrency == 1:
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(input_ids=inputs, attention_mask=attention_mask).logits
        _sync(torch, device)
        return [logits], [time.perf_counter() - started]

    if getattr(device, "type", str(device)) == "cuda":
        streams = [torch.cuda.Stream(device=device) for _ in range(concurrency)]
        started = time.perf_counter()
        outputs = []
        for stream in streams:
            with torch.cuda.stream(stream), torch.inference_mode():
                outputs.append(model(input_ids=inputs, attention_mask=attention_mask).logits)
        for stream in streams:
            stream.synchronize()
        # Requests enter each round together. Completion of the round is the
        # conservative end-to-end latency for every request in that group.
        elapsed = time.perf_counter() - started
        return outputs, [elapsed] * concurrency

    def infer() -> Any:
        with torch.inference_mode():
            return model(input_ids=inputs, attention_mask=attention_mask).logits

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        outputs = list(executor.map(lambda _: infer(), range(concurrency)))
    elapsed = time.perf_counter() - started
    return outputs, [elapsed] * concurrency


def _classification_quality_metrics(
    torch: Any,
    model: Any,
    model_name: str,
    model_revision: str | None,
    device: Any,
    sequence_length: int,
    sample_count: int,
) -> dict[str, int | float | str | bool | None]:
    """Evaluate DistilBERT precision runs on a fixed SST-2 validation slice."""
    if importlib.util.find_spec("datasets") is None:
        raise RuntimeError("Classification quality evaluation requires the datasets package")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if sample_count < 1:
        raise ValueError("quality_eval_samples must be at least 1")
    dataset = load_dataset("glue", "sst2", split=f"validation[:{sample_count}]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    encoded = tokenizer(
        list(dataset["sentence"]),
        padding=True,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in encoded.items()}
    labels = torch.tensor(dataset["label"], device=device)
    with torch.inference_mode():
        logits = model(**batch).logits
    _sync(torch, device)
    predictions = logits.argmax(dim=-1)
    accuracy = float((predictions == labels).float().mean().detach().cpu())
    return {
        "quality_accuracy": accuracy,
        "quality_samples": sample_count,
        "quality_dataset": "glue/sst2:validation",
        "quality_finite": bool(torch.isfinite(logits).all().item()),
    }


def _run_vllm_inference(config: ExperimentConfig) -> WorkloadRunOutput:
    parameters = config.workload.parameters
    serving_layout = declared_serving_layout(parameters)
    if (
        serving_layout == REPLICA_LAYOUT
        or int(parameters.get("data_parallel_size", 1)) > 1
        or int(parameters.get("replica_count", 1)) > 1
    ):
        raise UnsupportedWorkloadError(
            f"serving_layout={serving_layout} requires an executable multi-process replica router; "
            "the current vLLM adapter supports one engine process with tensor parallelism only"
        )
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vLLM framework requested, but vllm is not installed")
    import torch

    _prepare_runtime(torch, config, allow_distributed_strategy_none=True)
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM matrix experiments require CUDA")
    if config.runtime.precision not in {"fp16", "bf16", "fp8"}:
        raise RuntimeError(f"Generic vLLM adapter does not support precision: {config.runtime.precision}")

    quantization: str | None = None
    quantization_method: str | None = None
    base_compute_dtype: str | None = None
    if config.runtime.precision == "fp8":
        precision_backend = str(parameters.get("precision_backend", ""))
        quantization_method = str(parameters.get("quantization_method", ""))
        base_compute_dtype = str(parameters.get("base_compute_dtype", ""))
        if precision_backend != "vllm_fp8_online":
            raise RuntimeError("vLLM FP8 requires precision_backend=vllm_fp8_online")
        if quantization_method != "fp8_w8a8_dynamic":
            raise RuntimeError("vLLM FP8 requires quantization_method=fp8_w8a8_dynamic")
        if base_compute_dtype != "bf16":
            raise RuntimeError("vLLM FP8 requires base_compute_dtype=bf16")
        compute_capability = tuple(torch.cuda.get_device_capability(0))
        if compute_capability < (8, 9):
            raise RuntimeError(
                "vLLM FP8 W8A8 requires NVIDIA compute capability 8.9 or newer; "
                f"observed {compute_capability[0]}.{compute_capability[1]}"
            )
        dtype = "bfloat16"
        quantization = "fp8"
    else:
        dtype = "bfloat16" if config.runtime.precision == "bf16" else "float16"
        precision_backend = f"vllm_{dtype}"

    batch_size = int(parameters.get("batch_size", 1))
    concurrency = int(parameters.get("concurrent_requests", 1))
    prompt_length = int(parameters.get("prompt_length", 128))
    max_new_tokens = int(parameters.get("max_new_tokens", 32))
    _validate_context_window(config.workload.name, prompt_length, max_new_tokens)
    tensor_parallel_size = int(parameters.get("tensor_parallel_size", 1))
    if config.hardware.gpus > 1 and tensor_parallel_size != config.hardware.gpus:
        raise RuntimeError(
            f"tensor_parallel_size={tensor_parallel_size} does not match requested GPUs={config.hardware.gpus}"
        )
    torchdynamo_disabled = tensor_parallel_size > 1
    if torchdynamo_disabled:
        # vLLM 0.10.2 decorates its tensor-parallel vocabulary embedding
        # helper with ``torch.compile`` at module-import time. That direct
        # decorator bypasses vLLM's compilation_config, so TorchDynamo must be
        # disabled before importing vLLM in the certified TP runtime.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

    from vllm import LLM, SamplingParams

    model_name = str(parameters.get("hf_model_name", config.workload.model))
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "revision": _selected_model_revision(config, model_name),
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        # The certified private L40S runtime couples vLLM 0.10.2 to a
        # source-built PyTorch 2.8.0 wheel. Its installed Triton package is
        # sufficient for vLLM kernels, but not for vLLM's optional
        # torch.compile/Inductor path: that path expects Triton's removed
        # ``triton_key`` API. Eager execution avoids that incompatible optional
        # compiler while retaining CUDA, NCCL, and the vLLM engine itself.
        "enforce_eager": True,
    }
    if quantization is not None:
        llm_kwargs["quantization"] = quantization
    inductor_disabled = tensor_parallel_size > 1
    if inductor_disabled:
        # vLLM 0.10.2 preserves ``use_inductor=True`` in its configuration
        # even when ``enforce_eager=True`` selects compilation level zero. The
        # tensor-parallel vocabulary embedding path can still reach
        # TorchInductor in that state, which is incompatible with this pinned
        # Triton build. Disable Inductor explicitly for TP workloads.
        llm_kwargs["compilation_config"] = {"level": 0, "use_inductor": False}

    llm = LLM(**llm_kwargs)
    model_config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
    observed_quantization = getattr(model_config, "quantization", None)
    observed_base_dtype = str(getattr(model_config, "dtype", dtype))
    if quantization is not None and observed_quantization != quantization:
        raise RuntimeError(
            "vLLM did not activate the requested quantization backend: "
            f"requested {quantization}, observed {observed_quantization!r}"
        )
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    requests_per_round = batch_size * concurrency
    generator = torch.Generator().manual_seed(config.runtime.seed)
    prompt_token_ids = torch.randint(
        0,
        int(llm.get_tokenizer().vocab_size),
        (requests_per_round, prompt_length),
        generator=generator,
    ).tolist()
    prompts = [{"prompt_token_ids": token_ids} for token_ids in prompt_token_ids]
    measured_rounds = int(config.runtime.max_steps or 10)
    warmup_rounds = _warmup_steps(parameters, measured_rounds)
    for _ in range(warmup_rounds):
        llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    torch.cuda.synchronize()

    latencies: list[float] = []
    ttft_samples: list[float] = []
    queueing_samples: list[float] = []
    generated_tokens = 0
    nonempty_outputs = 0
    with measured_region():
        cuda_timer = _start_cuda_timer(torch)
        start_all = time.perf_counter()
        for _ in range(measured_rounds):
            start = time.perf_counter()
            outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            for output in outputs:
                output_tokens = sum(len(candidate.token_ids) for candidate in output.outputs)
                generated_tokens += output_tokens
                nonempty_outputs += int(output_tokens > 0)
                request_metrics = getattr(output, "metrics", None)
                arrival = getattr(request_metrics, "arrival_time", None)
                first = getattr(request_metrics, "first_token_time", None)
                finished = getattr(request_metrics, "finished_time", None)
                scheduled = getattr(request_metrics, "first_scheduled_time", None)
                if arrival is not None and first is not None:
                    ttft_samples.append(float(first - arrival))
                if arrival is not None and finished is not None:
                    latencies.append(float(finished - arrival))
                else:
                    latencies.append(elapsed)
                if arrival is not None and scheduled is not None:
                    queueing_samples.append(float(scheduled - arrival))
        cuda_event_seconds = _stop_cuda_timer(torch, cuda_timer)
        torch.cuda.synchronize()
        measured_seconds = max(time.perf_counter() - start_all, 1e-12)
    request_count = measured_rounds * requests_per_round
    metrics: dict[str, int | float | str | bool | None] = {
        "requests": request_count,
        "rounds": measured_rounds,
        "warmup_rounds": warmup_rounds,
        "batch_size": batch_size,
        "concurrent_requests": concurrency,
        "prompt_tokens_per_request": prompt_length,
        "generated_tokens": generated_tokens,
        "total_tokens": generated_tokens,
        "throughput_unit": "generated_tokens_per_second",
        "measured_seconds": measured_seconds,
        "gpu_seconds": measured_seconds * tensor_parallel_size,
        "gpu_hours": measured_seconds * tensor_parallel_size / 3600,
        "throughput_tokens_sec": generated_tokens / measured_seconds,
        "throughput_requests_sec": request_count / measured_seconds,
        "latency_p50_ms": _percentile(latencies, 50) * 1000,
        "latency_p95_ms": _percentile(latencies, 95) * 1000,
        "framework_observed": "vllm",
        "tensor_parallel_size_observed": tensor_parallel_size,
        "serving_layout_observed": serving_layout,
        "data_parallel_size_observed": 1,
        "replica_count_observed": 1,
        "vllm_execution_mode": "eager_no_inductor" if inductor_disabled else "eager",
        "vllm_inductor_disabled": inductor_disabled,
        "vllm_torchdynamo_disabled": torchdynamo_disabled,
        "precision_backend": precision_backend,
        "compute_dtype_observed": observed_base_dtype,
        "parameter_dtype_observed": (
            "mixed_fp8_linear_bf16_unquantized" if quantization is not None else observed_base_dtype
        ),
        "hf_model_name": model_name,
        "hf_model_revision": _selected_model_revision(config, model_name),
    }
    metrics.update(
        _serving_tail_metrics(
            latencies,
            ttft_samples,
            queueing_samples,
            measured_seconds,
            parameters.get("latency_slo_p95_ms"),
        )
    )
    if quantization is not None:
        metrics.update(
            {
                "quantization_method": quantization_method,
                "quantization_mode": "online",
                "vllm_quantization": observed_quantization,
                "quantized_linear_compute": "fp8_w8a8",
                "base_compute_dtype": base_compute_dtype,
                "comparison_control_id": str(parameters["comparison_control_id"]),
            }
        )
    if bool(parameters.get("quality_sanity_check", False)):
        metrics.update(
            {
                "quality_protocol": "fixed_synthetic_generation_sanity_v1",
                "quality_requests": request_count,
                "quality_nonempty_outputs": nonempty_outputs,
                "quality_request_success_rate": nonempty_outputs / max(request_count, 1),
                "quality_generated_tokens_positive": generated_tokens > 0,
            }
        )
    if cuda_event_seconds is not None:
        metrics["cuda_event_seconds"] = cuda_event_seconds
    if ttft_samples:
        metrics["time_to_first_token_p50_ms"] = _percentile(ttft_samples, 50) * 1000
        metrics["time_to_first_token_p95_ms"] = _percentile(ttft_samples, 95) * 1000
    if queueing_samples:
        metrics["queueing_time_p50_ms"] = _percentile(queueing_samples, 50) * 1000
        metrics["queueing_time_p95_ms"] = _percentile(queueing_samples, 95) * 1000
    return WorkloadRunOutput(
        metrics=metrics,
        samples={
            "latency_seconds": latencies,
            "time_to_first_token_seconds": ttft_samples,
            "queueing_time_seconds": queueing_samples,
        },
    )


def _prepare_runtime(
    torch: Any,
    config: ExperimentConfig,
    *,
    allow_distributed_strategy_none: bool = False,
) -> DistributedRuntime:
    requested_gpu_run = config.hardware.gpus > 0 and config.hardware.platform not in {"local", "mac_m1_dev"}
    if requested_gpu_run and not torch.cuda.is_available():
        raise RuntimeError(f"{config.experiment_id} requires CUDA; CPU/MPS fallback is forbidden for matrix GPU runs")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    expected_world_size = config.distributed.nodes * max(1, config.distributed.gpus_per_node)
    strategy = config.distributed.strategy
    needs_process_group = strategy != "none" and expected_world_size > 1
    if needs_process_group and world_size != expected_world_size:
        raise RuntimeError(
            f"{strategy} requires world size {expected_world_size}, observed {world_size}; launch with torchrun"
        )
    if strategy == "none" and world_size > 1 and not allow_distributed_strategy_none:
        raise RuntimeError("Multiple ranks were launched for a strategy=none experiment")
    if torch.cuda.is_available():
        if torch.cuda.device_count() < config.hardware.gpus:
            raise RuntimeError(
                f"Requested {config.hardware.gpus} GPUs, but only {torch.cuda.device_count()} CUDA devices are visible"
            )
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds visible CUDA device count {torch.cuda.device_count()}")
        torch.cuda.set_device(local_rank)
        requested_model = (config.hardware.gpu_model or "").lower()
        observed_model = str(torch.cuda.get_device_name(local_rank)).lower()
        required_token = "h100" if "h100" in requested_model else "l40s" if "l40s" in requested_model else ""
        if required_token and required_token not in observed_model:
            raise RuntimeError(
                f"Requested {config.hardware.gpu_model}, but CUDA device {local_rank} is "
                f"{torch.cuda.get_device_name(local_rank)}"
            )
    if needs_process_group:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend=config.distributed.backend, init_method="env://")
    _seed_everything(torch, config.runtime.seed, rank, deterministic=config.runtime.deterministic)
    return DistributedRuntime(rank=rank, local_rank=local_rank, world_size=world_size, distributed=needs_process_group)


def _configure_training_strategy(
    torch: Any,
    config: ExperimentConfig,
    runtime: DistributedRuntime,
    model: Any,
    device: Any,
    learning_rate: float,
    batch_size: int,
    accumulation_steps: int,
) -> tuple[Any, Any, Any | None, bool, dict[str, int | float | str | bool | None]]:
    strategy = config.distributed.strategy
    parameters = config.workload.parameters
    if strategy.startswith("deepspeed"):
        deepspeed_runtime = prepare_deepspeed_import_environment(torch)
        if importlib.util.find_spec("deepspeed") is None:
            raise RuntimeError(f"{strategy} requested, but DeepSpeed is not installed")
        try:
            import deepspeed
        except FileNotFoundError as exc:
            if "nvcc" in str(exc):
                raise RuntimeError(
                    f"{strategy} failed while DeepSpeed was probing nvcc. Source slurm/modules_for_hpc_ai before "
                    "running, or load a full CUDA toolkit whose CUDA_HOME/bin/nvcc exists. "
                    f"Prepared runtime details: {deepspeed_runtime}. Original error: {exc}"
                ) from exc
            raise
        deepspeed_nvtx_annotations_disabled = disable_deepspeed_nvtx_annotations()

        stage = int(strategy[-1])
        requested = parameters.get("deepspeed_config", {})
        if requested and int(requested.get("zero_optimization_stage", stage)) != stage:
            raise ValueError("DeepSpeed config stage does not match distributed.strategy")
        ds_config = {
            "train_micro_batch_size_per_gpu": batch_size,
            # The common loop performs accumulation explicitly so every
            # strategy uses identical global-work accounting.
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": stage},
            "bf16": {"enabled": config.runtime.precision == "bf16"},
            "fp16": {"enabled": config.runtime.precision == "fp16"},
        }
        if parameters.get("deepspeed_accumulation_mode") == "explicit_boundary_v1":
            ds_config["gradient_accumulation_steps"] = accumulation_steps
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=ds_config,
        )
        return engine, optimizer, engine, deepspeed_nvtx_annotations_disabled, {}

    model = model.to(device)
    strategy_metadata: dict[str, int | float | str | bool | None] = {}
    if strategy == "ddp" and runtime.world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        ddp_kwargs, comm_hook = _ddp_options(parameters)
        # Device ids are only valid for CUDA modules; CPU (gloo) smoke runs pass none.
        placement = (
            {"device_ids": [runtime.local_rank], "output_device": runtime.local_rank} if device.type == "cuda" else {}
        )
        model = DistributedDataParallel(model, **placement, **ddp_kwargs)
        if comm_hook != "none":
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

            model.register_comm_hook(state=None, hook=getattr(default_hooks, f"{comm_hook}_hook"))
        strategy_metadata = {
            "ddp_bucket_cap_mb_observed": ddp_kwargs.get("bucket_cap_mb", 25),
            "ddp_comm_hook_observed": comm_hook,
            "ddp_gradient_as_bucket_view_observed": ddp_kwargs.get("gradient_as_bucket_view", False),
            "ddp_static_graph_observed": ddp_kwargs.get("static_graph", False),
        }
    elif strategy == "fsdp" and runtime.world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy

        detail = str(parameters.get("strategy_detail", "full_shard"))
        sharding = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "sharded_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        }.get(detail)
        if sharding is None:
            raise ValueError(f"Unsupported FSDP sharding strategy: {detail}")
        mixed_precision = None
        if config.runtime.precision in {"fp16", "bf16"}:
            dtype = torch.float16 if config.runtime.precision == "fp16" else torch.bfloat16
            mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        fsdp_kwargs: dict[str, Any] = {
            "device_id": device,
            "sharding_strategy": sharding,
            "mixed_precision": mixed_precision,
            "use_orig_params": True,
        }
        if config.workload.name == "qwen2_5_7b_training":
            # A root-only FSDP wrapper all-gathers the entire 7B parameter
            # tree at once. On two H100 GPUs that creates a ~14 GiB transient
            # BF16 allocation on top of optimizer/gradient state. Wrap each
            # decoder layer so only one block is materialised at a time.
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            decoder_layer = _qwen_decoder_layer_class(model)
            fsdp_kwargs.update(
                {
                    "auto_wrap_policy": partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls={decoder_layer},
                    ),
                    "limit_all_gathers": True,
                    "forward_prefetch": False,
                }
            )
            strategy_metadata = {
                "fsdp_auto_wrap_policy": "qwen_decoder_layer",
                "fsdp_limit_all_gathers": True,
                "fsdp_forward_prefetch": False,
            }
        elif parameters.get("fsdp_wrap_policy") == "transformer_layer":
            # Shard and all-gather one transformer block at a time, so communication
            # can overlap compute; a root-only wrapper gathers the whole model at once.
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            fsdp_kwargs["auto_wrap_policy"] = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={_transformer_layer_class(model)},
            )
            strategy_metadata = {"fsdp_auto_wrap_policy": "transformer_layer"}
        else:
            strategy_metadata = {"fsdp_auto_wrap_policy": "root"}
        model = FullyShardedDataParallel(
            model,
            **fsdp_kwargs,
        )
    elif strategy not in {"none", "ddp", "fsdp"}:
        raise ValueError(f"Unsupported training strategy: {strategy}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer, None, False, strategy_metadata


def _ddp_options(parameters: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Translate validated DDP communication settings into constructor arguments.

    Only settings a config states explicitly are passed, so legacy DDP runs keep
    PyTorch's defaults (25 MB buckets, no hook) exactly as before.
    """
    kwargs: dict[str, Any] = {}
    if "ddp_bucket_cap_mb" in parameters:
        kwargs["bucket_cap_mb"] = int(parameters["ddp_bucket_cap_mb"])
    if "ddp_gradient_as_bucket_view" in parameters:
        kwargs["gradient_as_bucket_view"] = bool(parameters["ddp_gradient_as_bucket_view"])
    if "ddp_static_graph" in parameters:
        kwargs["static_graph"] = bool(parameters["ddp_static_graph"])
    return kwargs, str(parameters.get("ddp_comm_hook", "none"))


def _transformer_layer_class(model: Any) -> type[Any]:
    """Find the repeated block type for GPT-2, Llama-family and DistilBERT models."""
    for path in ("transformer.h", "model.layers", "distilbert.transformer.layer"):
        layers: Any = model
        for name in path.split("."):
            layers = getattr(layers, name, None)
            if layers is None:
                break
        if layers is not None and len(layers) > 0:
            return type(layers[0])
    raise RuntimeError(f"No known transformer layer stack on {type(model).__name__} for FSDP auto-wrap")


def _qwen_decoder_layer_class(model: Any) -> type[Any]:
    """Locate the concrete decoder-layer type for Qwen's FSDP auto-wrap.

    The model family is pinned by the matrix, but deriving the class from the
    loaded model avoids importing a Transformers-internal implementation path
    whose module location can vary across compatible releases.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Qwen2.5-7B FSDP requires a non-empty model.layers decoder stack")
    return type(layers[0])


def _training_precision(
    torch: Any,
    precision: str,
    device: Any,
    *,
    managed_by_deepspeed: bool = False,
    fp8_context: Any | None = None,
) -> tuple[Any, Any | None]:
    if managed_by_deepspeed:
        # DeepSpeed owns mixed precision and loss scaling through the bf16/fp16
        # section of its configuration. Nesting native AMP around its engine
        # produces an advisory warning and makes the observed backend unclear.
        return nullcontext, None
    if precision == "fp32":
        return nullcontext, None
    if precision == "fp8":
        if fp8_context is None:
            raise RuntimeError("FP8 training requires a configured Transformer Engine context")
        if getattr(device, "type", str(device)) != "cuda":
            raise RuntimeError("FP8 training requires a CUDA device")
        return fp8_context, None
    if precision not in {"fp16", "bf16"}:
        raise RuntimeError(
            f"Training precision {precision} requires a dedicated validated backend; generic substitution is forbidden"
        )
    if device.type != "cuda":
        return nullcontext, None
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16

    def context() -> Any:
        return torch.autocast(device_type="cuda", dtype=dtype)

    scaler = None
    if precision == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # pragma: no cover
            scaler = torch.cuda.amp.GradScaler()
    return context, scaler


def _uses_transformer_engine(config: ExperimentConfig) -> bool:
    return str(config.workload.parameters.get("precision_backend", "")) == "transformer_engine"


def _configure_transformer_engine_training(
    torch: Any,
    config: ExperimentConfig,
    model: Any,
    device: Any,
) -> tuple[Any, Any | None, dict[str, int | float | str | bool | None]]:
    """Replace eligible HF projections with matched Transformer Engine modules.

    Transformer Engine autocast only quantizes Transformer Engine modules. Merely
    wrapping an unmodified Hugging Face model would therefore produce mislabeled
    FP8 evidence. The same conversion is used for the BF16 controls so the
    controlled comparison changes precision without also changing module
    implementation. Unsupported output heads intentionally remain in BF16.
    """
    parameters = config.workload.parameters
    if parameters.get("precision_backend") != "transformer_engine":
        raise RuntimeError("Transformer Engine training requires precision_backend=transformer_engine")
    if config.runtime.precision not in {"bf16", "fp8"}:
        raise RuntimeError("Validated Transformer Engine training supports BF16 controls and FP8 treatments")
    if parameters.get("base_compute_dtype") != "bf16":
        raise RuntimeError("Transformer Engine training requires base_compute_dtype=bf16")
    if getattr(device, "type", str(device)) != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Transformer Engine training requires CUDA")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    fp8_enabled = config.runtime.precision == "fp8"
    if fp8_enabled and parameters.get("quantization_method") != "fp8_delayed_scaling_hybrid":
        raise RuntimeError("FP8 training requires quantization_method=fp8_delayed_scaling_hybrid")
    if fp8_enabled and capability < (8, 9):
        raise RuntimeError(
            "Transformer Engine FP8 training requires NVIDIA compute capability 8.9 or newer; "
            f"observed {capability[0]}.{capability[1]}"
        )
    if importlib.util.find_spec("transformer_engine.pytorch") is None:
        raise RuntimeError("Transformer Engine PyTorch support is not installed")

    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    model = model.to(dtype=torch.bfloat16)
    converted, skipped = _convert_projection_layers_to_transformer_engine(torch, te, model)
    if not converted:
        raise RuntimeError("Transformer Engine conversion found no FP8-compatible projection layers")

    context = None
    context_api = None
    if fp8_enabled:
        recipe = DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=int(parameters.get("fp8_amax_history_len", 16)),
            amax_compute_algo="max",
        )
        autocast = getattr(te, "autocast", None)
        if autocast is not None:

            def context() -> Any:
                return autocast(enabled=True, recipe=recipe)

            context_api = "transformer_engine.pytorch.autocast"
        else:  # Transformer Engine releases before the autocast rename.

            def context() -> Any:
                return te.fp8_autocast(enabled=True, fp8_recipe=recipe)

            context_api = "transformer_engine.pytorch.fp8_autocast"

    try:
        from importlib.metadata import version

        te_version = version("transformer-engine")
    except Exception:  # pragma: no cover - unusual source/editable installations
        te_version = str(getattr(te, "__version__", "unknown"))
    metadata: dict[str, int | float | str | bool | None] = {
        "transformer_engine_version": te_version,
        "transformer_engine_module_count": len(converted),
        "transformer_engine_modules": ",".join(converted),
        "transformer_engine_skipped_projection_count": len(skipped),
        "transformer_engine_skipped_projections": ",".join(skipped),
        "precision_backend": (
            "transformer_engine_delayed_scaling_hybrid" if fp8_enabled else "transformer_engine_bf16"
        ),
        "compute_dtype_observed": "fp8_mixed_bf16" if fp8_enabled else "bf16",
        "fp8_recipe": "delayed_scaling_hybrid" if fp8_enabled else None,
        "fp8_amax_history_len": int(parameters.get("fp8_amax_history_len", 16)) if fp8_enabled else None,
        "fp8_autocast_api": context_api,
        "fp8_compute_capability": f"{capability[0]}.{capability[1]}",
        "fp8_forward_enabled": fp8_enabled,
    }
    return model, context, metadata


def _convert_projection_layers_to_transformer_engine(
    torch: Any,
    te: Any,
    model: Any,
) -> tuple[list[str], list[str]]:
    """Convert shape-compatible encoder/decoder projections and preserve weights.

    Sequence-classification heads operate on pooled ``[batch, hidden]``
    activations. A small diagnostic batch can violate Transformer Engine's FP8
    requirement that the flattened leading dimension is divisible by eight, so
    those task heads deliberately remain BF16.
    """
    converted: list[str] = []
    skipped: list[str] = []
    task_head_names = {"classifier", "pre_classifier", "score"}

    def visit(parent: Any, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            is_linear = isinstance(child, torch.nn.Linear)
            is_hf_conv1d = type(child).__name__ == "Conv1D" and type(child).__module__.startswith("transformers")
            if not (is_linear or is_hf_conv1d):
                visit(child, qualified)
                continue

            if name in task_head_names:
                skipped.append(qualified)
                continue

            if is_linear:
                in_features = int(child.in_features)
                out_features = int(child.out_features)
                source_weight = child.weight.detach()
            else:
                in_features = int(child.weight.shape[0])
                out_features = int(child.weight.shape[1])
                source_weight = child.weight.detach().transpose(0, 1)
            if in_features % 16 or out_features % 16:
                skipped.append(qualified)
                continue

            replacement = te.Linear(
                in_features,
                out_features,
                bias=child.bias is not None,
                params_dtype=torch.bfloat16,
            )
            replacement = replacement.to(device=child.weight.device, dtype=torch.bfloat16)
            with torch.no_grad():
                replacement.weight.copy_(source_weight.to(replacement.weight))
                replacement.weight.requires_grad_(child.weight.requires_grad)
                if child.bias is not None and replacement.bias is not None:
                    replacement.bias.copy_(child.bias.detach().to(replacement.bias))
                    replacement.bias.requires_grad_(child.bias.requires_grad)
            replacement.train(child.training)
            setattr(parent, name, replacement)
            converted.append(qualified)

    visit(model)
    return converted, skipped


def _enable_activation_checkpointing(model: Any, *, prefer_non_reentrant: bool) -> str:
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError(f"{type(model).__name__} does not support activation checkpointing")
    if prefer_non_reentrant:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            return "hf_gradient_checkpointing_non_reentrant"
        except TypeError:
            model.gradient_checkpointing_enable()
            return "hf_gradient_checkpointing_default_kwargs_unavailable"
    model.gradient_checkpointing_enable()
    return "hf_gradient_checkpointing_default"


def _inference_dtype(torch: Any, precision: str, device: Any) -> Any:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if precision == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    raise RuntimeError(f"Inference precision {precision} requires a dedicated quantized/FP8 backend")


def _training_data_iterator(
    torch: Any,
    config: ExperimentConfig,
    runtime: DistributedRuntime,
    *,
    task: str,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    num_labels: int,
    required_batches: int,
) -> Any | None:
    parameters = config.workload.parameters
    if "data_movement_study_type" not in parameters and "num_workers" not in parameters:
        return None
    dataset = _SyntheticTokenDataset(
        length=max(required_batches * batch_size * runtime.world_size, batch_size * runtime.world_size),
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        num_labels=num_labels,
        task=task,
        seed=config.runtime.seed,
    )
    sampler = None
    if runtime.world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            shuffle=False,
        )
    workers = int(parameters.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(parameters.get("pinned_memory", False)),
        "drop_last": True,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(parameters.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = True
    return iter(torch.utils.data.DataLoader(**kwargs))


class _SyntheticTokenDataset:
    """Deterministic CPU-resident samples for controlled DataLoader experiments."""

    def __init__(
        self,
        *,
        length: int,
        sequence_length: int,
        vocab_size: int,
        num_labels: int,
        task: str,
        seed: int,
    ) -> None:
        self.length = length
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.task = task
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        generator = torch.Generator().manual_seed(self.seed + index)
        input_ids = torch.randint(0, self.vocab_size, (self.sequence_length,), generator=generator)
        if self.task == "causal_lm":
            return {"input_ids": input_ids, "labels": input_ids.clone()}
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.randint(0, self.num_labels, (), generator=generator),
        }


def _make_device_batch(
    torch: Any,
    task: str,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    num_labels: int,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        device=device,
    )
    if task == "causal_lm":
        return {"input_ids": input_ids, "labels": input_ids}
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.randint(0, num_labels, (batch_size,), generator=generator, device=device),
    }


def _benchmark_allreduce(
    torch: Any,
    runtime: DistributedRuntime,
    device: Any,
    parameters: dict[str, Any],
) -> dict[str, int | float | str | bool | None]:
    import torch.distributed as dist

    tensor_bytes = int(parameters.get("allreduce_bytes", 64 * 1024 * 1024))
    iterations = int(parameters.get("allreduce_iterations", 10))
    tensor = torch.ones(max(1, tensor_bytes // 4), dtype=torch.float32, device=device)
    for _ in range(3):
        dist.all_reduce(tensor)
    _sync(torch, device)
    start = time.perf_counter()
    for _ in range(iterations):
        dist.all_reduce(tensor)
    _sync(torch, device)
    elapsed = max(time.perf_counter() - start, 1e-12)
    elapsed = _distributed_max(torch, elapsed, device, runtime)
    algorithmic_bytes = 2 * (runtime.world_size - 1) / runtime.world_size * tensor_bytes * iterations
    return {
        "allreduce_time_seconds": elapsed / iterations,
        "nccl_bandwidth_gbps": algorithmic_bytes / elapsed / 1e9,
        "allreduce_payload_bytes": tensor_bytes,
        "allreduce_iterations": iterations,
    }


def _selected_framework(config: ExperimentConfig) -> str:
    parameters = config.workload.parameters
    framework = parameters.get("framework")
    if framework:
        return str(framework)
    frameworks = parameters.get("frameworks")
    if isinstance(frameworks, list):
        if "pytorch" in frameworks or "pytorch_eager" in frameworks:
            return "pytorch_eager"
        if len(frameworks) == 1:
            return str(frameworks[0])
    return "pytorch_eager"


def _selected_model_revision(config: ExperimentConfig, model_name: str) -> str | None:
    parameters = config.workload.parameters
    override_revision = parameters.get("hf_model_revision")
    if override_revision:
        return str(override_revision)
    if model_name == config.workload.model:
        return config.workload.model_revision
    return None


def _strategy_detail(config: ExperimentConfig) -> str:
    parameters = config.workload.parameters
    if config.distributed.strategy == "fsdp":
        return str(
            parameters.get("strategy_detail")
            or parameters.get("fsdp_config", {}).get("sharding_strategy")
            or "full_shard"
        )
    if config.distributed.strategy.startswith("deepspeed_zero"):
        return f"zero_stage_{config.distributed.strategy[-1]}"
    return config.distributed.strategy


def _per_gpu_batch_size(parameters: dict[str, Any]) -> int:
    value = parameters.get("micro_batch_size", parameters.get("per_gpu_batch_size", parameters.get("batch_size", 1)))
    batch_size = int(value)
    if batch_size < 1:
        raise ValueError("Per-GPU batch size must be at least 1")
    return batch_size


def _validate_batch_semantics(
    config: ExperimentConfig,
    runtime: DistributedRuntime,
    batch_size: int,
    accumulation_steps: int,
) -> None:
    parameters = config.workload.parameters
    observed_global = batch_size * accumulation_steps * runtime.world_size
    declared_effective = parameters.get("effective_batch_size")
    if declared_effective is not None and int(declared_effective) != observed_global:
        raise ValueError(f"effective_batch_size={declared_effective}, but runtime implies {observed_global}")
    declared_global = parameters.get("global_batch_size")
    if declared_global is not None and int(declared_global) != observed_global:
        raise ValueError(f"global_batch_size={declared_global}, but runtime implies {observed_global}")


def _model_config(model: Any) -> Any:
    wrapped = getattr(model, "module", model)
    return wrapped.config


def _distributed_mean(torch: Any, value: float, device: Any, runtime: DistributedRuntime) -> float:
    if runtime.world_size == 1:
        return value
    import torch.distributed as dist

    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    tensor /= runtime.world_size
    return float(tensor.cpu())


def _distributed_max(torch: Any, value: float, device: Any, runtime: DistributedRuntime) -> float:
    """Use the slowest rank for globally valid distributed throughput."""
    if runtime.world_size == 1:
        return value
    import torch.distributed as dist

    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


def _warmup_steps(parameters: dict[str, Any], measured_steps: int) -> int:
    if "warmup_steps" in parameters:
        return max(0, int(parameters["warmup_steps"]))
    if measured_steps <= 1:
        return 0
    return min(5, max(1, measured_steps // 10))


def _seed_everything(torch: Any, seed: int, rank: int, *, deterministic: bool) -> None:
    import random

    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic


def _sync(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)


def _start_cuda_timer(torch: Any) -> tuple[Any, Any] | None:
    """Record the start of a CUDA-event timing region on the current device."""
    if not torch.cuda.is_available():
        return None
    started = torch.cuda.Event(enable_timing=True)
    completed = torch.cuda.Event(enable_timing=True)
    started.record()
    return started, completed


def _stop_cuda_timer(torch: Any, timer: tuple[Any, Any] | None) -> float | None:
    """Finish a CUDA-event region and return elapsed device time in seconds."""
    if timer is None:
        return None
    started, completed = timer
    completed.record()
    completed.synchronize()
    return max(float(started.elapsed_time(completed)) / 1000, 0.0)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _serving_tail_metrics(
    latencies: list[float],
    ttft_samples: list[float],
    queueing_samples: list[float],
    measured_seconds: float,
    latency_slo_p95_ms: object,
) -> dict[str, int | float | str | bool | None]:
    """Return tail metrics shared by eager and vLLM serving adapters.

    Latency samples are seconds while the public metrics are milliseconds. A
    configured SLO is optional; when present, goodput counts requests whose
    observed end-to-end latency is within that SLO. No SLO is inferred from a
    benchmark configuration because that would turn a performance probe into
    an undocumented service objective.
    """
    metrics: dict[str, int | float | str | bool | None] = {}
    if latencies:
        metrics.update(
            {
                "latency_p50_ms": _percentile(latencies, 50) * 1000,
                "latency_p95_ms": _percentile(latencies, 95) * 1000,
                "latency_p99_ms": _percentile(latencies, 99) * 1000,
            }
        )
    if ttft_samples:
        metrics.update(
            {
                "time_to_first_token_p50_ms": _percentile(ttft_samples, 50) * 1000,
                "time_to_first_token_p95_ms": _percentile(ttft_samples, 95) * 1000,
                "time_to_first_token_p99_ms": _percentile(ttft_samples, 99) * 1000,
            }
        )
    if queueing_samples:
        metrics.update(
            {
                "queueing_time_p50_ms": _percentile(queueing_samples, 50) * 1000,
                "queueing_time_p95_ms": _percentile(queueing_samples, 95) * 1000,
                "queueing_time_p99_ms": _percentile(queueing_samples, 99) * 1000,
            }
        )
    if latency_slo_p95_ms is not None:
        try:
            slo = float(latency_slo_p95_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("latency_slo_p95_ms must be a positive finite number") from exc
        if not math.isfinite(slo) or slo <= 0:
            raise ValueError("latency_slo_p95_ms must be a positive finite number")
        if not latencies:
            raise RuntimeError("latency_slo_p95_ms requires at least one measured latency sample")
        met = sum(sample * 1000 <= slo for sample in latencies)
        metrics.update(
            {
                "latency_slo_p95_ms": slo,
                "goodput_requests_sec": met / max(measured_seconds, 1e-12),
                "goodput_slo_met_requests": met,
                "goodput_slo_met_fraction": met / len(latencies),
            }
        )
    return metrics


def _precision_backend(precision: str, *, managed_by_deepspeed: bool = False) -> str:
    if managed_by_deepspeed:
        return {
            "fp32": "deepspeed_fp32",
            "fp16": "deepspeed_fp16",
            "bf16": "deepspeed_bf16",
        }.get(precision, f"deepspeed_unsupported_{precision}")
    return {
        "fp32": "pytorch_fp32",
        "fp16": "pytorch_amp_fp16",
        "bf16": "pytorch_amp_bf16",
        "fp8": "transformer_engine_delayed_scaling_hybrid",
    }.get(precision, f"unsupported_{precision}")


def _validate_context_window(workload: str, input_tokens: int, generated_tokens: int) -> None:
    limits = {"distilbert": 512, "gpt2_small": 1024, "tinyllama": 2048}
    if input_tokens < 1 or generated_tokens < 0:
        raise ValueError("Token lengths must be positive")
    limit = limits.get(workload)
    required = input_tokens + generated_tokens
    if limit is not None and required > limit:
        raise ValueError(
            f"{workload} requires {required} positions, exceeding its configured context window of {limit}"
        )


def _load_causal_lm_stack() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    return torch, AutoModelForCausalLM


def _load_sequence_classification_stack() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification

    return torch, AutoModelForSequenceClassification


# Backward-compatible helpers retained for local tests and integrations.
def _device(torch: Any, parameters: dict[str, Any] | None = None) -> Any:
    requested = str((parameters or {}).get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available")
    if requested == "mps" and not _mps_available(torch):
        raise RuntimeError("Requested MPS device, but Apple MPS is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device setting: {requested}")
    return torch.device(requested)


def _mps_available(torch: Any) -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend and backend.is_available())


def _torch_dtype(torch: Any, precision: str, device: Any) -> Any:
    return _inference_dtype(torch, precision, device)


def _run_causal_lm_inference(config: ExperimentConfig) -> WorkloadRunOutput:
    return _run_causal_inference(config)
