"""Pinned, versioned contract for the low-bit serving extension.

This is a comparison of publisher-released artifacts, not a claim that AWQ and
GPTQ were calibrated identically. Never reinterpret the older FP8/INT8 rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

PROTOCOL = "qwen25_7b_lowbit_v1"
QUALITY_PROTOCOL = "qwen25_7b_lowbit_quality_v2"
QUALITY_METHOD_IDS = {"bf16": "E270", "awq": "E271", "gptq": "E272", "bitsandbytes_nf4": "E273"}
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
ARTIFACTS = {
    "bf16": (BASE_MODEL, BASE_REVISION, None),
    "awq": (
        "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "b25037543e9394b818fdfca67ab2a00ecc7dd641",
        "awq_marlin",
    ),
    "gptq": (
        "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "e9c932ac1893a49ae0fc497ad6e1e86e2e39af20",
        "gptq_marlin",
    ),
    "bitsandbytes_nf4": (BASE_MODEL, BASE_REVISION, "bitsandbytes"),
}
METHOD_IDS = {
    "bf16": ("E239", "E240"),
    "awq": ("E241", "E242"),
    "gptq": ("E243", "E244"),
    "bitsandbytes_nf4": ("E245", "E246"),
}
PARAMETERS = {
    "study_protocol",
    "quantization_method",
    "framework",
    "base_compute_dtype",
    "batch_size",
    "prompt_length",
    "max_new_tokens",
    "warmup_steps",
    "min_measured_seconds",
    "prompt_samples",
    "quality_samples",
    "quality_sequence_length",
    "quality_max_relative_ppl_increase",
    "dataset_revision",
    "dataset_subset",
    "performance_split",
    "quality_split",
    "max_model_len",
    "kv_cache_blocks",
    "kv_cache_block_size",
    "gpu_memory_utilization",
    "comparison_control_id",
}


def lowbit_issues(config: Any) -> list[str]:
    p = config.workload.parameters
    issues = []
    if config.mode != "inference" or config.hardware.gpus != 1 or config.distributed.strategy != "none":
        issues.append("low-bit serving requires single-GPU inference with strategy=none")
    if config.distributed.nodes != 1 or config.distributed.gpus_per_node != 1:
        issues.append("low-bit serving requires one node and one GPU per node")
    if p.get("study_protocol") not in {PROTOCOL, QUALITY_PROTOCOL} or p.get("framework") != "vllm":
        issues.append("low-bit serving requires the versioned vLLM study protocol")
    method = p.get("quantization_method")
    if method not in ARTIFACTS:
        return issues + ["unsupported low-bit quantization method"]
    model, revision, _ = ARTIFACTS[method]
    if (config.workload.model, config.workload.model_revision) != (model, revision):
        issues.append("low-bit model artifact must match its pinned publisher revision")
    expected_precision = {"bf16": "bf16", "bitsandbytes_nf4": "nf4"}.get(method, "int4")
    if config.runtime.precision != expected_precision:
        issues.append("low-bit precision must distinguish BF16, integer INT4, and bitsandbytes NF4")
    if p.get("base_compute_dtype") != "bf16":
        issues.append("low-bit serving holds activation/compute dtype at BF16")
    if (config.workload.dataset, p.get("dataset_revision")) != (DATASET, DATASET_REVISION):
        issues.append("low-bit dataset must use the pinned WikiText revision")
    if (p.get("dataset_subset"), p.get("performance_split"), p.get("quality_split")) != (
        "wikitext-2-raw-v1",
        "validation",
        "test",
    ):
        issues.append("low-bit performance/quality must use separate WikiText validation/test splits")
    positive = (
        "batch_size",
        "prompt_length",
        "max_new_tokens",
        "warmup_steps",
        "min_measured_seconds",
        "prompt_samples",
        "quality_samples",
        "quality_sequence_length",
        "max_model_len",
        "kv_cache_blocks",
        "kv_cache_block_size",
    )
    for key in positive:
        value = p.get(key)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or (key != "min_measured_seconds" and not isinstance(value, int))
        ):
            issues.append(f"low-bit {key} must be positive")
    if issues:
        return issues
    if p["batch_size"] not in {1, 8} or p["prompt_samples"] < p["batch_size"]:
        issues.append("low-bit batch size must be 1 or 8, with enough distinct prompts")
    if not 0 < p.get("gpu_memory_utilization", 0) <= 0.9:
        issues.append("low-bit gpu_memory_utilization must be in (0, 0.9]")
    if not 0 <= p.get("quality_max_relative_ppl_increase", -1) <= 1:
        issues.append("low-bit quality tolerance must be preregistered in [0, 1]")
    if max(p["prompt_length"] + p["max_new_tokens"], p["quality_sequence_length"] + 1) > p["max_model_len"]:
        issues.append("low-bit inputs exceed max_model_len")
    if p["max_model_len"] > 32768 or p["quality_sequence_length"] < 2:
        issues.append("low-bit lengths exceed the model contract or cannot produce quality logprobs")
    if p["batch_size"] * (p["prompt_length"] + p["max_new_tokens"]) > p["kv_cache_blocks"] * p["kv_cache_block_size"]:
        issues.append("low-bit fixed KV cache is too small for this request batch")
    quality_extension = p.get("study_protocol") == QUALITY_PROTOCOL
    if quality_extension and (p["batch_size"], p["quality_samples"], p["quality_sequence_length"]) != (1, 128, 512):
        issues.append("quality v2 requires batch 1 and 128 distinct 512-token test windows")
    control = "E270" if quality_extension else ("E239" if p["batch_size"] == 1 else "E240")
    if p.get("comparison_control_id") != control:
        issues.append("low-bit comparison must reference the BF16 control at the same batch size")
    if not config.runtime.max_steps or config.runtime.max_steps < 2:
        issues.append("low-bit max_steps must allow at least two measured rounds")
    return issues


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_fingerprint() -> str:
    """Invalidate gates only when the actual runtime/contract changes, not docs."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "config/lowbit.py",
        "workloads/lowbit.py",
        "workloads/transformer.py",
        "runners/local.py",
        "profiling/system.py",
        "profiling/measurement.py",
    )
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths})
