"""Fail-closed validation for the H100 Qwen2.5-7B sharding canaries."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import config_sha256, discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config

EXPECTATIONS = {
    "E250": {"strategy": "fsdp", "detail": "full_shard"},
    "E251": {"strategy": "deepspeed_zero3", "detail": "zero_stage_3"},
}


def verify_canaries(result_root: Path, platform: str = "h100") -> dict[str, dict[str, Any]]:
    """Validate current, collected canary evidence and return a small summary."""

    if platform != "h100":
        raise ValueError("Qwen2.5-7B training canaries are currently H100-only")
    expected_configs = resolved_canary_configs(platform)
    issues: list[str] = []
    accepted: dict[str, dict[str, Any]] = {}
    for experiment_id, expectation in EXPECTATIONS.items():
        expected = expected_configs[experiment_id]
        expected_hash = canary_contract_sha256(expected)
        records = matching_records(result_root / experiment_id, expected_hash)
        if not records:
            issues.append(f"{experiment_id}: no collected record matches the current resolved canary configuration")
            continue
        record_path, record = records[-1]
        metrics = record.get("metrics", {})
        environment = record.get("environment", {})
        prefix = f"{experiment_id}: "
        checks = {
            "completed": record.get("status") == "completed" and record.get("error") is None,
            "condition": record.get("condition_id") == f"{experiment_id}@{platform}",
            "workload": record.get("workload") == "qwen2_5_7b_training",
            "BF16 precision": record.get("precision") == "bf16",
            "two GPUs": record.get("gpus") == 2,
            "H100 runtime": "h100" in str(environment.get("observed_gpu_model", "")).lower(),
            "clean repository": environment.get("git_dirty") is False,
            "strategy": metrics.get("strategy_observed") == expectation["strategy"],
            "strategy detail": metrics.get("strategy_detail_observed") == expectation["detail"],
            "BF16 checkpoint load": metrics.get("model_load_dtype_requested") == "bf16",
            "effective batch": metrics.get("effective_global_batch_size") == 16,
            "sequence length": metrics.get("sequence_length") == 512,
            "activation checkpointing": metrics.get("activation_checkpointing") is True,
            "positive throughput": positive_finite(metrics.get("throughput_tokens_sec")),
            "finite loss": finite(metrics.get("loss")),
            "finite held-out loss": metrics.get("quality_finite") is True and finite(metrics.get("quality_loss")),
            "memory measurement": positive_finite(metrics.get("memory_used_gb")),
        }
        if experiment_id == "E250":
            checks["layer-wise FSDP wrapping"] = (
                metrics.get("fsdp_auto_wrap_policy") == "qwen_decoder_layer"
                and metrics.get("fsdp_limit_all_gathers") is True
                and metrics.get("fsdp_forward_prefetch") is False
            )
        issues.extend(prefix + label + " failed" for label, passed in checks.items() if not passed)
        accepted[experiment_id] = {
            "result": str(record_path),
            "canary_contract_sha256": expected_hash,
            "strategy": metrics.get("strategy_observed"),
            "strategy_detail": metrics.get("strategy_detail_observed"),
            "throughput_tokens_sec": metrics.get("throughput_tokens_sec"),
            "memory_used_gb": metrics.get("memory_used_gb"),
        }
    if issues:
        raise ValueError("Qwen2.5-7B sharding canary failed:\n- " + "\n- ".join(issues))
    return accepted


def resolved_canary_configs(platform: str) -> dict[str, Any]:
    entries = {
        entry.config.experiment_id: entry for entry in discover_experiment_configs("configs/large_model_training")
    }
    profile = load_platform_profile(f"platforms/{platform}.json")
    conditions = {
        condition.matrix_id: condition
        for condition in load_platform_conditions(
            "platforms/execution_matrix.json", platform, wave="qwen7b-training-canary", include_deferred=False
        )
    }
    return {
        experiment_id: resolve_platform_config(entries[experiment_id], profile, conditions[experiment_id])
        for experiment_id in EXPECTATIONS
    }


def matching_records(root: Path, expected_hash: str) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record.get("config")
        if isinstance(config, dict) and canary_contract_sha256(config) == expected_hash:
            matches.append((path, record))
    return matches


def canary_contract_sha256(config: Any) -> str:
    """Hash a resolved config without its bundle-local source file location.

    Each submission bundle correctly records the snapshot path from which it
    was rendered. That absolute path changes per bundle and is therefore not a
    portable runtime contract. The source-file checksum remains in the
    fingerprint, so this normalisation does not weaken provenance checks.
    """

    payload = config.to_dict() if hasattr(config, "to_dict") else deepcopy(dict(config))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("source_config_path", None)
    return config_sha256(payload)


def finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def positive_finite(value: object) -> bool:
    return finite(value) and float(value) > 0
