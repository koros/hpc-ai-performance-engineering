"""Real-text, fixed-cache Qwen2.5-7B long-context serving measurements."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from importlib.metadata import version
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.config.long_context import PROTOCOL, long_context_issues, runtime_fingerprint
from hpc_ai_perf.config.lowbit import BASE_MODEL, BASE_REVISION, digest
from hpc_ai_perf.profiling import measured_region
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput
from hpc_ai_perf.workloads.lowbit import (
    _worker_quantization,
    load_artifact_config,
    request_measurements,
    runtime_packages,
    select_prompts,
    text_windows,
    validate_artifact,
    worker_memory,
)
from hpc_ai_perf.workloads.transformer import _percentile, _prepare_runtime


class QwenLongContextWorkload(Workload):
    name = "qwen2_5_7b_long_context"

    def prepare(self, config: ExperimentConfig) -> None:
        issues = long_context_issues(config)
        if issues:
            raise ValueError("; ".join(issues))
        required = ("torch", "vllm", "transformers", "datasets", "huggingface_hub")
        missing = [package for package in required if importlib.util.find_spec(package) is None]
        if missing:
            raise RuntimeError("Long-context environment is missing: " + ", ".join(missing))
        if version("vllm").split("+")[0] != "0.10.2":
            raise RuntimeError("Long-context protocol requires vLLM 0.10.2; recertify before changing engines")
        if os.environ.get("VLLM_USE_V1", "0") != "0":
            raise RuntimeError("Long-context protocol requires VLLM_USE_V1=0 for request timestamps")
        os.environ["VLLM_USE_V1"] = "0"

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        _prepare_runtime(torch, config)
        if not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability(0)) < (8, 9):
            raise RuntimeError("Long-context study requires an allocated L40S/H100-class CUDA GPU")
        p = config.workload.parameters
        artifact_config = load_artifact_config(config)
        validate_artifact("bf16", artifact_config)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        data = load_dataset(
            config.workload.dataset,
            p["dataset_subset"],
            revision=p["dataset_revision"],
            split=p["performance_split"],
        )
        tokens = text_windows(data["text"], tokenizer, p["prompt_length"], p["prompt_samples"])
        llm = LLM(**engine_kwargs(config))
        observed = llm.llm_engine.model_config
        if observed.quantization is not None or str(observed.dtype) != "torch.bfloat16":
            raise RuntimeError(f"Requested unquantized BF16; observed {observed.quantization}/{observed.dtype}")
        if observed.max_model_len != p["max_model_len"]:
            raise RuntimeError("Engine did not retain the fixed maximum context length")
        cache = llm.llm_engine.cache_config
        if cache.num_gpu_blocks != p["kv_cache_blocks"] or cache.block_size != p["kv_cache_block_size"]:
            raise RuntimeError("Engine did not retain the fixed KV-cache allocation")
        backend_details = llm.llm_engine.model_executor.collective_rpc(_worker_quantization)[0]
        if backend_details["quantized_module_count"] != 0:
            raise RuntimeError("The long-context BF16 study unexpectedly loaded quantized modules")
        sampling = SamplingParams(temperature=0, max_tokens=p["max_new_tokens"], ignore_eos=True)
        for step in range(p["warmup_steps"]):
            llm.generate(select_prompts(tokens, step, 1), sampling, use_tqdm=False)
        before_memory = worker_memory(llm, reset=True)
        torch.cuda.synchronize()
        rounds = generated = 0
        latencies: list[float] = []
        ttfts: list[float] = []
        with measured_region():
            started = time.perf_counter()
            while rounds < int(config.runtime.max_steps or 0):
                outputs = llm.generate(select_prompts(tokens, rounds, 1), sampling, use_tqdm=False)
                count, latency, ttft = request_measurements(outputs, 1, p["max_new_tokens"])
                generated += count
                latencies.extend(latency)
                ttfts.extend(ttft)
                rounds += 1
                if rounds >= 2 and time.perf_counter() - started >= p["min_measured_seconds"]:
                    break
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
        if seconds < p["min_measured_seconds"]:
            raise RuntimeError("Measured interval too short; increase max_steps instead of accepting sparse telemetry")
        after_memory = worker_memory(llm)
        return WorkloadRunOutput(
            metrics={
                "study_protocol": PROTOCOL,
                "long_context_runtime_sha256": runtime_fingerprint(),
                "long_context_parameters_sha256": digest(p),
                "runtime_packages": json.dumps(runtime_packages(), sort_keys=True),
                "framework_observed": "vllm",
                "vllm_execution_mode": "v0_eager",
                "compute_dtype_observed": str(observed.dtype),
                "vllm_quantization": observed.quantization,
                "base_model": BASE_MODEL,
                "base_model_revision": BASE_REVISION,
                "hf_model_name": config.workload.model,
                "hf_model_revision": config.workload.model_revision,
                "artifact_config_sha256": digest(artifact_config),
                "dataset_revision": p["dataset_revision"],
                "performance_tokens_sha256": digest(tokens),
                "prompt_source": "wikitext_validation_eos_separated_stream_v1",
                "request_protocol": "offline_closed_batch_not_http_serving",
                "batch_size": 1,
                "concurrent_requests": 1,
                "prompt_tokens_per_request": p["prompt_length"],
                "max_new_tokens": p["max_new_tokens"],
                "requests": rounds,
                "rounds": rounds,
                "warmup_rounds": p["warmup_steps"],
                "prefill_tokens": rounds * p["prompt_length"],
                "generated_tokens": generated,
                "total_tokens": generated,
                "throughput_unit": "generated_tokens_per_second",
                "throughput_tokens_sec": generated / seconds,
                "throughput_requests_sec": rounds / seconds,
                "measured_seconds": seconds,
                "gpu_seconds": seconds,
                "gpu_hours": seconds / 3600,
                "latency_p50_ms": _percentile(latencies, 50) * 1000,
                "latency_p95_ms": _percentile(latencies, 95) * 1000,
                "latency_p99_ms": _percentile(latencies, 99) * 1000,
                "time_to_first_token_p50_ms": _percentile(ttfts, 50) * 1000,
                "time_to_first_token_p95_ms": _percentile(ttfts, 95) * 1000,
                "time_to_first_token_p99_ms": _percentile(ttfts, 99) * 1000,
                "latency_sample_count": len(latencies),
                "max_model_len_observed": observed.max_model_len,
                "kv_cache_blocks_observed": cache.num_gpu_blocks,
                "kv_cache_block_size_observed": cache.block_size,
                "kv_cache_dtype_observed": "bfloat16",
                "memory_model_weights_gb": before_memory["model_weights_bytes"] / 1024**3,
                "memory_worker_peak_allocated_gb": after_memory["peak_allocated_bytes"] / 1024**3,
                "memory_worker_peak_reserved_gb": after_memory["peak_reserved_bytes"] / 1024**3,
                "memory_scope": "worker_allocator_including_fixed_kv_cache;weights_from_vllm_loader",
            },
            samples={"latency_seconds": latencies, "time_to_first_token_seconds": ttfts},
        )


def engine_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    p = config.workload.parameters
    return {
        "model": config.workload.model,
        "revision": config.workload.model_revision,
        "tokenizer": BASE_MODEL,
        "tokenizer_revision": BASE_REVISION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "seed": config.runtime.seed,
        "enforce_eager": True,
        "trust_remote_code": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "max_model_len": p["max_model_len"],
        "max_num_seqs": 1,
        "max_num_batched_tokens": p["max_model_len"],
        "gpu_memory_utilization": p["gpu_memory_utilization"],
        "num_gpu_blocks_override": p["kv_cache_blocks"],
        "block_size": p["kv_cache_block_size"],
        "kv_cache_dtype": "auto",
        "swap_space": 0,
        "cpu_offload_gb": 0,
    }
