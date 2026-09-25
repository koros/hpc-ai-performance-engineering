"""Controlled offline serving and real-text quality evaluation, separate from legacy trials."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.config.lowbit import (
    ARTIFACTS,
    BASE_MODEL,
    BASE_REVISION,
    digest,
    lowbit_issues,
    runtime_fingerprint,
)
from hpc_ai_perf.profiling import measured_region
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.transformer import _percentile, _prepare_runtime


class LowBitServingWorkload(Workload):
    name = "qwen2_5_7b"

    def prepare(self, config: ExperimentConfig) -> None:
        issues = lowbit_issues(config)
        if issues:
            raise ValueError("; ".join(issues))
        required = ["torch", "vllm", "transformers", "datasets"]
        if config.workload.parameters["quantization_method"] == "bitsandbytes_nf4":
            required.append("bitsandbytes")
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise RuntimeError("Low-bit environment is missing: " + ", ".join(missing))
        if version("vllm").split("+")[0] != "0.10.2":
            raise RuntimeError("Low-bit v1 protocol requires vLLM 0.10.2; recertify before changing engines")
        if "bitsandbytes" in required and version("bitsandbytes") != "0.47.0":
            raise RuntimeError("The NF4 arm requires bitsandbytes==0.47.0")
        # V0 provides per-request timestamps. Do not substitute batch wall time
        # for TTFT or request latency if a different engine omits them.
        if os.environ.get("VLLM_USE_V1", "0") != "0":
            raise RuntimeError("Low-bit v1 protocol requires VLLM_USE_V1=0")
        os.environ["VLLM_USE_V1"] = "0"

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        _prepare_runtime(torch, config)
        if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) < (8, 9):
            raise RuntimeError("Low-bit study requires an allocated L40S/H100-class CUDA GPU")
        p = config.workload.parameters
        method = str(p["quantization_method"])
        artifact_config = load_artifact_config(config)
        validate_artifact(method, artifact_config)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        perf_data = load_dataset(
            config.workload.dataset, p["dataset_subset"], revision=p["dataset_revision"], split=p["performance_split"]
        )
        quality_data = load_dataset(
            config.workload.dataset, p["dataset_subset"], revision=p["dataset_revision"], split=p["quality_split"]
        )
        perf_tokens = text_windows(perf_data["text"], tokenizer, p["prompt_length"], p["prompt_samples"])
        quality_tokens = text_windows(
            quality_data["text"], tokenizer, p["quality_sequence_length"], p["quality_samples"]
        )
        llm = LLM(**engine_kwargs(config))
        observed = llm.llm_engine.model_config
        expected_backend = ARTIFACTS[method][2]
        if observed.quantization != expected_backend or str(observed.dtype) != "torch.bfloat16":
            raise RuntimeError(f"Requested {expected_backend}/BF16; observed {observed.quantization}/{observed.dtype}")
        cache = llm.llm_engine.cache_config
        if cache.num_gpu_blocks != p["kv_cache_blocks"] or cache.block_size != p["kv_cache_block_size"]:
            raise RuntimeError("Engine did not retain the fixed KV-cache allocation")
        sampling = SamplingParams(temperature=0, max_tokens=p["max_new_tokens"], ignore_eos=True)
        batch_size = p["batch_size"]
        for step in range(p["warmup_steps"]):
            llm.generate(select_prompts(perf_tokens, step, batch_size), sampling, use_tqdm=False)
        # Evaluation is outside the throughput/energy interval and uses a
        # separate split. Unknown upstream calibration overlap is documented.
        quality = quality_metrics(llm, SamplingParams, quality_tokens)
        before_memory = worker_memory(llm, reset=True)
        backend_details = llm.llm_engine.model_executor.collective_rpc(_worker_quantization)[0]
        validate_backend(method, backend_details)
        torch.cuda.synchronize()
        generated = rounds = 0
        latencies: list[float] = []
        ttfts: list[float] = []
        with measured_region():
            started = time.perf_counter()
            while rounds < int(config.runtime.max_steps or 0):
                outputs = llm.generate(select_prompts(perf_tokens, rounds, batch_size), sampling, use_tqdm=False)
                count, latency, ttft = request_measurements(outputs, batch_size, p["max_new_tokens"])
                generated += count
                latencies.extend(latency)
                ttfts.extend(ttft)
                rounds += 1
                if rounds >= 2 and time.perf_counter() - started >= p["min_measured_seconds"]:
                    break
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        if seconds < p["min_measured_seconds"]:
            raise RuntimeError(
                "Measured interval too short; increase max_steps instead of trusting sparse energy samples"
            )
        after_memory = worker_memory(llm)
        metrics = {
            "study_protocol": p["study_protocol"],
            "study_runtime_sha256": runtime_fingerprint(),
            "study_parameters_sha256": digest(p),
            "runtime_packages": json.dumps(runtime_packages(), sort_keys=True),
            "comparison_scope": "publisher_artifact_deployment",
            "comparison_control_id": p["comparison_control_id"],
            "upstream_calibration_provenance": "not_fully_disclosed" if method in {"awq", "gptq"} else "not_applicable",
            "quantization_method": method,
            "precision_backend": expected_backend or "vllm_bf16",
            "vllm_quantization": observed.quantization,
            "compute_dtype_observed": str(observed.dtype),
            "framework_observed": "vllm",
            "vllm_execution_mode": "v0_eager",
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_REVISION,
            "hf_model_name": config.workload.model,
            "hf_model_revision": config.workload.model_revision,
            "artifact_config_sha256": digest(artifact_config),
            "artifact_quantization_config": json.dumps(artifact_config.get("quantization_config", {}), sort_keys=True),
            "quantized_module_count": backend_details["quantized_module_count"],
            "quantized_module_types": json.dumps(backend_details["quantized_module_types"]),
            "bnb_quant_types_observed": json.dumps(backend_details["bnb_quant_types"]),
            "dataset_revision": p["dataset_revision"],
            "performance_tokens_sha256": digest(perf_tokens),
            "quality_tokens_sha256": digest(quality_tokens),
            "request_protocol": "offline_closed_batch_not_http_serving",
            "batch_size": batch_size,
            "requests": rounds * batch_size,
            "rounds": rounds,
            "warmup_rounds": p["warmup_steps"],
            "generated_tokens": generated,
            "total_tokens": generated,
            "throughput_unit": "generated_tokens_per_second",
            "throughput_tokens_sec": generated / seconds,
            "throughput_requests_sec": rounds * batch_size / seconds,
            "measured_seconds": seconds,
            "gpu_seconds": seconds,
            "gpu_hours": seconds / 3600,
            "latency_p50_ms": _percentile(latencies, 50) * 1000,
            "latency_p95_ms": _percentile(latencies, 95) * 1000,
            "time_to_first_token_p50_ms": _percentile(ttfts, 50) * 1000,
            "time_to_first_token_p95_ms": _percentile(ttfts, 95) * 1000,
            "latency_sample_count": len(latencies),
            "kv_cache_blocks_observed": cache.num_gpu_blocks,
            "kv_cache_block_size_observed": cache.block_size,
            "kv_cache_dtype_observed": "bfloat16",  # kv_cache_dtype=auto with BF16 model dtype
            "memory_model_weights_gb": before_memory["model_weights_bytes"] / 1024**3,
            "memory_worker_peak_allocated_gb": after_memory["peak_allocated_bytes"] / 1024**3,
            "memory_worker_peak_reserved_gb": after_memory["peak_reserved_bytes"] / 1024**3,
            "memory_scope": "worker_allocator_including_fixed_kv_cache;weights_from_vllm_loader",
            **quality,
        }
        return WorkloadRunOutput(metrics=metrics)


def engine_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    p = config.workload.parameters
    backend = ARTIFACTS[p["quantization_method"]][2]
    kwargs = {
        "model": config.workload.model,
        "revision": config.workload.model_revision,
        "tokenizer": BASE_MODEL,
        "tokenizer_revision": BASE_REVISION,
        "dtype": "bfloat16",
        "quantization": backend,
        "tensor_parallel_size": 1,
        "seed": config.runtime.seed,
        "enforce_eager": True,
        "trust_remote_code": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_model_len": p["max_model_len"],
        "max_num_seqs": 8,
        "max_num_batched_tokens": p["max_model_len"] * 8,
        "gpu_memory_utilization": p["gpu_memory_utilization"],
        "num_gpu_blocks_override": p["kv_cache_blocks"],
        "block_size": p["kv_cache_block_size"],
        "kv_cache_dtype": "auto",
        "swap_space": 0,
        "cpu_offload_gb": 0,
    }
    if backend == "bitsandbytes":
        kwargs["load_format"] = "bitsandbytes"
    return kwargs


def validate_artifact(method: str, config: dict[str, Any]) -> None:
    if config.get("model_type") != "qwen2" or config.get("num_hidden_layers") != 28:
        raise RuntimeError("Artifact is not the expected Qwen2.5-7B architecture")
    quant = config.get("quantization_config") or {}
    if method in {"awq", "gptq"}:
        if (quant.get("quant_method"), quant.get("bits"), quant.get("group_size")) != (method, 4, 128):
            raise RuntimeError("Artifact quantization metadata does not match the 4-bit/group-128 contract")
    elif quant:
        raise RuntimeError("BF16 and in-flight NF4 require the unquantized base artifact")


def load_artifact_config(config: ExperimentConfig) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(config.workload.model, "config.json", revision=config.workload.model_revision)
    return json.loads(Path(path).read_text())


def text_windows(texts: Any, tokenizer: Any, length: int, count: int) -> list[list[int]]:
    """Non-overlapping token-stream windows with EOS separators between corpus rows.

    WikiText rows are short paragraphs, not documents. Concatenation is explicit
    and identical for every artifact; quality and performance splits stay apart.
    """
    windows = []
    pending: list[int] = []
    if not isinstance(tokenizer.eos_token_id, int):
        raise RuntimeError("Pinned tokenizer must provide an EOS row separator")
    for text in texts:
        if not text.strip():
            continue
        pending.extend(tokenizer.encode(text, add_special_tokens=False))
        pending.append(tokenizer.eos_token_id)
        while len(pending) >= length:
            windows.append(pending[:length])
            pending = pending[length:]
            if len(windows) == count:
                return windows
    raise RuntimeError(f"Dataset yielded {len(windows)} windows, fewer than the required {count}")


def select_prompts(tokens: list[list[int]], step: int, batch_size: int) -> list[dict[str, list[int]]]:
    return [{"prompt_token_ids": tokens[(step * batch_size + index) % len(tokens)]} for index in range(batch_size)]


def quality_metrics(llm: Any, sampling_class: Any, tokens: list[list[int]]) -> dict[str, Any]:
    nll = 0.0
    scored = 0
    # Single-document requests keep evaluation shape identical across load arms.
    sampling = sampling_class(temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=1)
    for window in tokens:
        outputs = llm.generate([{"prompt_token_ids": window}], sampling, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].prompt_logprobs or []) != len(window):
            raise RuntimeError("Quality evaluation requires complete prompt-token log probabilities")
        for token, entries in zip(window[1:], outputs[0].prompt_logprobs[1:], strict=True):
            entry = (entries or {}).get(token)
            value = getattr(entry, "logprob", None)
            if not isinstance(value, int | float) or not math.isfinite(value) or value > 0:
                raise RuntimeError("Missing/non-finite target-token log probability in quality evaluation")
            nll -= value
            scored += 1
    loss = nll / scored
    ppl = math.exp(loss)
    if not math.isfinite(ppl):
        raise RuntimeError("Quality perplexity is not finite")
    return {
        "quality_protocol": "wikitext_test_eos_separated_stream_token_nll_v1",
        "quality_loss": loss,
        "quality_perplexity": ppl,
        "quality_tokens": scored,
        "quality_samples": len(tokens),
        "quality_finite": True,
        "quality_scope": "text_likelihood_not_instruction_accuracy;upstream_calibration_overlap_unknown",
    }


def request_measurements(
    outputs: Any, expected_requests: int, expected_tokens: int
) -> tuple[int, list[float], list[float]]:
    if len(outputs) != expected_requests:
        raise RuntimeError("Engine returned an incomplete request batch")
    latencies, ttfts = [], []
    for output in outputs:
        if len(output.outputs) != 1 or len(output.outputs[0].token_ids) != expected_tokens:
            raise RuntimeError("Engine did not generate the fixed number of tokens")
        metrics = output.metrics
        arrival = getattr(metrics, "arrival_time", None)
        first = getattr(metrics, "first_token_time", None)
        finished = getattr(metrics, "finished_time", None)
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in (arrival, first, finished)):
            raise RuntimeError("Per-request latency/TTFT timestamps unavailable; no batch-time fallback is permitted")
        if not arrival <= first <= finished or finished <= arrival:
            raise RuntimeError("Invalid request timestamp ordering")
        latencies.append(finished - arrival)
        ttfts.append(first - arrival)
    return expected_requests * expected_tokens, latencies, ttfts


def _worker_memory_snapshot(worker: Any, reset: bool = False) -> dict[str, Any]:
    """Executed in the vLLM worker, not the mostly empty parent CUDA process."""
    import torch

    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {
        "model_weights_bytes": worker.model_runner.model_memory_usage,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def worker_memory(llm: Any, *, reset: bool = False) -> dict[str, Any]:
    rows = llm.llm_engine.model_executor.collective_rpc(_worker_memory_snapshot, kwargs={"reset": reset})
    if len(rows) != 1 or any(
        not isinstance(value, int | float) or not math.isfinite(value) or value <= 0 for value in rows[0].values()
    ):
        raise RuntimeError("Could not measure memory in the single vLLM worker")
    return rows[0]


def runtime_packages() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError

    packages = {}
    for package in ("torch", "vllm", "transformers", "tokenizers", "datasets", "huggingface-hub", "bitsandbytes"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return packages


def _worker_quantization(worker: Any) -> dict[str, Any]:
    names = []
    types = set()
    for module in worker.model_runner.model.modules():
        name = type(getattr(module, "quant_method", None)).__name__
        if name in {"AWQMarlinLinearMethod", "GPTQMarlinLinearMethod", "BitsAndBytesLinearMethod"}:
            names.append(name)
        for param in module.parameters(recurse=False):
            for state in getattr(param, "bnb_quant_state", {}).values():
                types.add(state.quant_type)
    return {
        "quantized_module_count": len(names),
        "quantized_module_types": sorted(set(names)),
        "bnb_quant_types": sorted(types),
    }


def validate_backend(method: str, observed: dict[str, Any]) -> None:
    expected = {
        "awq": "AWQMarlinLinearMethod",
        "gptq": "GPTQMarlinLinearMethod",
        "bitsandbytes_nf4": "BitsAndBytesLinearMethod",
    }
    if method == "bf16":
        if observed["quantized_module_count"] != 0:
            raise RuntimeError("BF16 control unexpectedly contains quantized modules")
    elif observed["quantized_module_count"] < 1 or observed["quantized_module_types"] != [expected[method]]:
        raise RuntimeError("Requested quantized linear implementation was not observed in the worker")
    if method == "bitsandbytes_nf4" and observed["bnb_quant_types"] != ["nf4"]:
        raise RuntimeError("bitsandbytes did not activate NF4 weights")
