from __future__ import annotations

import importlib.util
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.runtime_checks import disable_deepspeed_nvtx_annotations, prepare_deepspeed_import_environment
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


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


def transformer_workloads() -> list[TransformerWorkload]:
    return [GPT2SmallWorkload(), DistilBERTWorkload(), TinyLlamaWorkload()]


def _run_causal_training(config: ExperimentConfig) -> WorkloadRunOutput:
    torch, auto_model = _load_causal_lm_stack()
    runtime = _prepare_runtime(torch, config)
    parameters = config.workload.parameters
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    model = auto_model.from_pretrained(
        model_name,
        revision=_selected_model_revision(config, model_name),
        torch_dtype=torch.float32,
    )
    return _run_training_loop(torch, config, runtime, model, task="causal_lm", model_name=model_name)


def _run_classification_training(config: ExperimentConfig) -> WorkloadRunOutput:
    torch, auto_model = _load_sequence_classification_stack()
    runtime = _prepare_runtime(torch, config)
    parameters = config.workload.parameters
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    num_labels = int(parameters.get("num_labels", 2))
    model = auto_model.from_pretrained(
        model_name,
        revision=_selected_model_revision(config, model_name),
        num_labels=num_labels,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
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
    warmup_steps = _warmup_steps(parameters, steps)
    batch_size = _per_gpu_batch_size(parameters)
    sequence_length = int(parameters.get("sequence_length", 128))
    accumulation_steps = int(parameters.get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    _validate_batch_semantics(config, runtime, batch_size, accumulation_steps)
    learning_rate = float(parameters.get("learning_rate", 5e-5))
    device = torch.device("cuda", runtime.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    activation_checkpointing_backend = "disabled"
    if bool(parameters.get("activation_checkpointing", False)):
        activation_checkpointing_backend = _enable_activation_checkpointing(
            model,
            prefer_non_reentrant=config.distributed.strategy == "fsdp",
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    model, optimizer, engine, deepspeed_nvtx_annotations_disabled = _configure_training_strategy(
        torch,
        config,
        runtime,
        model,
        device,
        learning_rate,
        batch_size,
        accumulation_steps,
    )
    deepspeed_managed_precision = engine is not None
    amp_context, scaler = _training_precision(
        torch,
        config.runtime.precision,
        device,
        managed_by_deepspeed=deepspeed_managed_precision,
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
    if data_iterator is None:
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    data_loading_seconds = 0.0
    transfer_seconds = 0.0
    final_loss = 0.0

    def one_optimizer_step(measured: bool) -> float:
        nonlocal data_loading_seconds, transfer_seconds
        if engine is not None:
            engine.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_load = 0.0
        step_transfer = 0.0
        for accumulation_index in range(accumulation_steps):
            if data_iterator is None:
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
                with amp_context():
                    outputs = model(**batch)
                    loss = outputs.loss / accumulation_steps
                if engine is not None:
                    engine.backward(loss)
                elif scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                step_loss += float(loss.detach().cpu())

        if engine is not None:
            engine.step()
        elif scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if measured:
            data_loading_seconds += step_load
            transfer_seconds += step_transfer
        return step_loss

    model.train()
    for _ in range(warmup_steps):
        one_optimizer_step(False)
    _sync(torch, device)

    step_times: list[float] = []
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
        "compute_dtype_observed": config.runtime.precision,
        "parameter_dtype_observed": str(next(model.parameters()).dtype),
        "dataset_execution": "cpu_dataloader" if data_iterator is not None else "synthetic_device_tensors",
        "deepspeed_nvtx_annotations_disabled": deepspeed_nvtx_annotations_disabled,
    }
    if cuda_event_seconds is not None:
        metrics["cuda_event_seconds"] = cuda_event_seconds
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
    return WorkloadRunOutput(metrics=metrics)


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
    cuda_timer = _start_cuda_timer(torch)
    start_all = time.perf_counter()
    logits = None
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
    cuda_event_seconds = _stop_cuda_timer(torch, cuda_timer)
    measured_seconds = max(time.perf_counter() - start_all, 1e-12)
    total_samples = measured_batches * batch_size * concurrency
    total_tokens = total_samples * sequence_length
    metrics: dict[str, int | float | str | bool | None] = {
        "requests": total_samples,
        "samples": total_samples,
        "batches": measured_batches,
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
    return WorkloadRunOutput(metrics=metrics)


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
    if importlib.util.find_spec("vllm") is None:
        raise RuntimeError("vLLM framework requested, but vllm is not installed")
    import torch

    _prepare_runtime(torch, config, allow_distributed_strategy_none=True)
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM matrix experiments require CUDA")
    if config.runtime.precision not in {"fp16", "bf16"}:
        raise RuntimeError(f"Generic vLLM adapter does not support precision: {config.runtime.precision}")

    parameters = config.workload.parameters
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
    dtype = "bfloat16" if config.runtime.precision == "bf16" else "float16"
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
    inductor_disabled = tensor_parallel_size > 1
    if inductor_disabled:
        # vLLM 0.10.2 preserves ``use_inductor=True`` in its configuration
        # even when ``enforce_eager=True`` selects compilation level zero. The
        # tensor-parallel vocabulary embedding path can still reach
        # TorchInductor in that state, which is incompatible with this pinned
        # Triton build. Disable Inductor explicitly for TP workloads.
        llm_kwargs["compilation_config"] = {"level": 0, "use_inductor": False}

    llm = LLM(**llm_kwargs)
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

    latencies: list[float] = []
    ttft_samples: list[float] = []
    queueing_samples: list[float] = []
    generated_tokens = 0
    cuda_timer = _start_cuda_timer(torch)
    start_all = time.perf_counter()
    for _ in range(measured_rounds):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        elapsed = time.perf_counter() - start
        for output in outputs:
            generated_tokens += sum(len(candidate.token_ids) for candidate in output.outputs)
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
        "vllm_execution_mode": "eager_no_inductor" if inductor_disabled else "eager",
        "vllm_inductor_disabled": inductor_disabled,
        "vllm_torchdynamo_disabled": torchdynamo_disabled,
        "precision_backend": f"vllm_{dtype}",
        "compute_dtype_observed": dtype,
        "parameter_dtype_observed": dtype,
        "hf_model_name": model_name,
        "hf_model_revision": _selected_model_revision(config, model_name),
    }
    if cuda_event_seconds is not None:
        metrics["cuda_event_seconds"] = cuda_event_seconds
    if ttft_samples:
        metrics["time_to_first_token_p50_ms"] = _percentile(ttft_samples, 50) * 1000
        metrics["time_to_first_token_p95_ms"] = _percentile(ttft_samples, 95) * 1000
    if queueing_samples:
        metrics["queueing_time_p50_ms"] = _percentile(queueing_samples, 50) * 1000
        metrics["queueing_time_p95_ms"] = _percentile(queueing_samples, 95) * 1000
    return WorkloadRunOutput(metrics=metrics)


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
) -> tuple[Any, Any, Any | None, bool]:
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
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            config=ds_config,
        )
        return engine, optimizer, engine, deepspeed_nvtx_annotations_disabled

    model = model.to(device)
    if strategy == "ddp" and runtime.world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, device_ids=[runtime.local_rank], output_device=runtime.local_rank)
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
        model = FullyShardedDataParallel(
            model,
            device_id=device,
            sharding_strategy=sharding,
            mixed_precision=mixed_precision,
            use_orig_params=True,
        )
    elif strategy not in {"none", "ddp", "fsdp"}:
        raise ValueError(f"Unsupported training strategy: {strategy}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return model, optimizer, None, False


def _training_precision(
    torch: Any,
    precision: str,
    device: Any,
    *,
    managed_by_deepspeed: bool = False,
) -> tuple[Any, Any | None]:
    if managed_by_deepspeed:
        # DeepSpeed owns mixed precision and loss scaling through the bf16/fp16
        # section of its configuration. Nesting native AMP around its engine
        # produces an advisory warning and makes the observed backend unclear.
        return nullcontext, None
    if precision == "fp32":
        return nullcontext, None
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
