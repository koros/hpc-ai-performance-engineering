"""Pinned contract for the Qwen2.5-7B long-context serving continuation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from hpc_ai_perf.config.lowbit import BASE_MODEL, BASE_REVISION, DATASET, DATASET_REVISION, digest

PROTOCOL = "qwen25_7b_long_context_v1"
CONTEXT_IDS = {4096: "E247", 6144: "E248", 8192: "E249"}
MAX_MODEL_LEN = 9216
KV_CACHE_BLOCKS = 1024
KV_CACHE_BLOCK_SIZE = 16
PARAMETERS = {
    "study_protocol",
    "framework",
    "base_compute_dtype",
    "batch_size",
    "concurrent_requests",
    "prompt_length",
    "max_new_tokens",
    "warmup_steps",
    "min_measured_seconds",
    "prompt_samples",
    "dataset_revision",
    "dataset_subset",
    "performance_split",
    "max_model_len",
    "kv_cache_blocks",
    "kv_cache_block_size",
    "gpu_memory_utilization",
}


def long_context_issues(config: Any) -> list[str]:
    """Reject any shape drift before it reaches a GPU reservation."""
    p = config.workload.parameters
    issues = []
    if config.mode != "inference" or config.hardware.gpus != 1 or config.distributed.strategy != "none":
        issues.append("long-context serving requires single-GPU inference with strategy=none")
    if config.distributed.nodes != 1 or config.distributed.gpus_per_node != 1:
        issues.append("long-context serving requires one node and one GPU per node")
    if p.get("study_protocol") != PROTOCOL or p.get("framework") != "vllm":
        issues.append("long-context serving requires the versioned vLLM study protocol")
    if (config.workload.model, config.workload.model_revision) != (BASE_MODEL, BASE_REVISION):
        issues.append("long-context serving must use the pinned Qwen2.5-7B BF16 artifact")
    if config.runtime.precision != "bf16" or p.get("base_compute_dtype") != "bf16":
        issues.append("long-context serving holds model and activation compute at BF16")
    if (config.workload.dataset, p.get("dataset_revision")) != (DATASET, DATASET_REVISION):
        issues.append("long-context serving must use the pinned WikiText revision")
    if (p.get("dataset_subset"), p.get("performance_split")) != ("wikitext-2-raw-v1", "validation"):
        issues.append("long-context serving must use the pinned WikiText validation split")
    positive_ints = (
        "batch_size",
        "concurrent_requests",
        "prompt_length",
        "max_new_tokens",
        "warmup_steps",
        "prompt_samples",
        "max_model_len",
        "kv_cache_blocks",
        "kv_cache_block_size",
    )
    for key in positive_ints:
        value = p.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            issues.append(f"long-context {key} must be a positive integer")
    duration = p.get("min_measured_seconds")
    if (
        not isinstance(duration, int | float)
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration <= 0
    ):
        issues.append("long-context min_measured_seconds must be positive")
    if issues:
        return issues
    if p["prompt_length"] not in CONTEXT_IDS or config.experiment_id != CONTEXT_IDS[p["prompt_length"]]:
        issues.append("long-context prompt length must use its registered E247-E249 condition")
    if p["batch_size"] != 1 or p["concurrent_requests"] != 1 or p["prompt_samples"] < 8:
        issues.append("long-context study fixes one request at a time with at least eight prompts")
    if p["max_new_tokens"] != 128:
        issues.append("long-context study fixes generation to 128 tokens")
    if p["warmup_steps"] != 3 or p["min_measured_seconds"] < 30:
        issues.append("long-context study requires three warm-ups and at least 30 measured seconds")
    if p["max_model_len"] != MAX_MODEL_LEN:
        issues.append("long-context study fixes max_model_len to 9216 across prompt lengths")
    if (p["kv_cache_blocks"], p["kv_cache_block_size"]) != (KV_CACHE_BLOCKS, KV_CACHE_BLOCK_SIZE):
        issues.append("long-context study fixes the BF16 KV-cache allocation across prompt lengths")
    if p["prompt_length"] + p["max_new_tokens"] > p["max_model_len"]:
        issues.append("long-context input plus generation exceeds max_model_len")
    if p["batch_size"] * (p["prompt_length"] + p["max_new_tokens"]) > p["kv_cache_blocks"] * p["kv_cache_block_size"]:
        issues.append("long-context fixed KV cache is too small for the request")
    if not 0 < p.get("gpu_memory_utilization", 0) <= 0.9:
        issues.append("long-context gpu_memory_utilization must be in (0, 0.9]")
    if not config.runtime.max_steps or config.runtime.max_steps < 2:
        issues.append("long-context max_steps must allow at least two measured rounds")
    return issues


def runtime_fingerprint() -> str:
    """Fingerprint executable sources only; documentation does not invalidate a gate."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "config/long_context.py",
        "config/lowbit.py",
        "workloads/long_context.py",
        "workloads/lowbit.py",
        "runners/local.py",
        "profiling/system.py",
        "profiling/measurement.py",
    )
    payload = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    return digest(payload)
