"""Collected-evidence gate for the scaling deep-dive performance waves."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.config.scaling_deep_dive import CAPACITY_GB, MEMORY_RULE, design
from hpc_ai_perf.validation import validate_result_file
from hpc_ai_perf.validation.qwen7b_training import canary_contract_sha256
from hpc_ai_perf.validation.scaling_acceptance import _manifest_entries
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256

# Audited training compatibility, deliberately bound to exact source hashes.
# fa85a91d -> 2f86c47 changes only inference definitions in transformer.py;
# the other three fingerprinted files and existing training definitions match.
# Keep the recorded fingerprints intact. Any subsequent runtime edit requires
# fresh canaries or another explicit source audit, even for inference changes.
_AUDITED_CANARY_RUNTIMES = {
    "2b3d8e4d0c702d0def2b6ff8defc6eda5f19d106d8946e4c98b043ed2e479bec": frozenset(
        {"c3485b1978e83f4175ebd27d76f9a80216d06c5bbf729fe959c6421b09d1b8b8"}
    ),
}


def _compatible_canary_runtime(observed: object) -> bool:
    current = scaling_runtime_sha256()
    return isinstance(observed, str) and (observed == current or observed in _AUDITED_CANARY_RUNTIMES.get(current, ()))


def verify_deep_dive_canaries(
    result_root: str | Path,
    platform: str,
    *,
    execution_matrix: str | Path = "platforms/execution_matrix.json",
) -> dict[str, str]:
    """Require every planned platform canary to be collected and within its measured memory ceiling."""
    if platform not in CAPACITY_GB:
        raise ValueError("Scaling deep-dive platform must be h100 or l40s")
    root = Path(result_root)
    checksums = _manifest_entries(root / "SHA256SUMS")
    conditions = load_platform_conditions(execution_matrix, platform, wave="scaling-dd-canary")
    expected_ids = {
        eid for eid, condition, _ in design() if condition.role == "canary" and platform in condition.run_platforms
    }
    if {condition.matrix_id for condition in conditions} != expected_ids or any(
        condition.required_trials != 1 for condition in conditions
    ):
        raise ValueError(f"{platform}: scaling deep-dive canary policy is incomplete or has incorrect trial counts")
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs("configs/scaling_deep_dive")}
    profile = load_platform_profile(f"platforms/{platform}.json")
    accepted: dict[str, str] = {}
    issues: list[str] = []
    for condition in conditions:
        eid = condition.matrix_id
        entry = entries.get(eid)
        if entry is None:
            issues.append(f"{eid}: missing canary configuration")
            continue
        config = resolve_platform_config(entry, profile, condition)
        if config.workload.parameters.get("scaling_role") != "canary":
            issues.append(f"{eid}: matrix entry is not a canary")
            continue
        expected_contract = canary_contract_sha256(config)
        matched = False
        measurement_failures: set[str] = set()
        for path in sorted((root / eid).glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
                continue
            if canary_contract_sha256(record["config"]) != expected_contract:
                continue
            relative = path.relative_to(root).as_posix()
            if checksums.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
                issues.append(f"{eid}: matching result is absent from or mismatches SHA256SUMS")
                continue
            if validate_result_file(path):
                issues.append(f"{eid}: matching result fails evidence validation")
                continue
            failed_checks = _canary_issues(record, config, condition.condition_id, platform)
            if not failed_checks:
                accepted[eid] = str(path)
                matched = True
                break
            measurement_failures.update(failed_checks)
        if not matched:
            detail = ""
            if measurement_failures:
                detail = "; matching evidence failed: " + ", ".join(sorted(measurement_failures))
            issues.append(f"{eid}: no completed, collected canary passes the current contract and measurements{detail}")
    if issues:
        raise ValueError("Scaling deep-dive acceptance failed:\n- " + "\n- ".join(issues))
    return accepted


def _canary_issues(record: dict[str, Any], config: Any, condition_id: str, platform: str) -> list[str]:
    metrics = record.get("metrics") or {}
    environment = record.get("environment") or {}
    p = config.workload.parameters
    gpus = config.hardware.gpus
    updates = p["warmup_steps"] + config.runtime.max_steps
    memory = metrics.get("memory_used_gb")
    checks = {
        "completed": record.get("status") == "completed" and record.get("error") is None,
        "platform and condition": record.get("platform_id") == platform and record.get("condition_id") == condition_id,
        "clean source": record.get("git_dirty") is False and environment.get("git_dirty") is False,
        "GPU model": platform in str(environment.get("observed_gpu_model", "")).lower(),
        "GPU count": record.get("gpus") == gpus and metrics.get("world_size") == gpus,
        "runtime fingerprint": _compatible_canary_runtime(metrics.get("scaling_runtime_sha256")),
        "input protocol": metrics.get("dataset_execution") == p["input_protocol"],
        "input digest": isinstance(metrics.get("global_input_sha256"), str)
        and len(metrics["global_input_sha256"]) == 64
        and all(char in "0123456789abcdef" for char in metrics["global_input_sha256"]),
        "update accounting": metrics.get("optimizer_step_calls") == updates
        and metrics.get("expected_optimizer_step_calls") == updates
        and metrics.get("backward_calls") == updates * p["gradient_accumulation_steps"],
        "batch accounting": metrics.get("effective_global_batch_size") == p["global_batch_size"],
        "strategy": metrics.get("strategy_observed") == config.distributed.strategy,
        "compute precision": metrics.get("compute_dtype_observed")
        == ("fp8_mixed_bf16" if config.runtime.precision == "fp8" else config.runtime.precision),
        "attention backend": metrics.get("attention_backend_observed") == "sdpa"
        and metrics.get("training_use_cache") is False,
        "throughput": _positive(metrics.get("throughput_tokens_sec")),
        "loss": _finite(metrics.get("initial_training_loss")) and _finite(metrics.get("loss")),
        "memory ceiling": _positive(memory) and memory < CAPACITY_GB[platform] * MEMORY_RULE,
    }
    if p.get("precision_backend") == "transformer_engine":
        backend = (
            "transformer_engine_delayed_scaling_hybrid"
            if config.runtime.precision == "fp8"
            else "transformer_engine_bf16"
        )
        checks["Transformer Engine"] = metrics.get("precision_backend") == backend
        if config.runtime.precision == "fp8":
            checks["FP8 execution"] = metrics.get("fp8_forward_enabled") is True
    if config.distributed.strategy == "ddp" and gpus > 1:
        checks["DDP communication hook"] = metrics.get("ddp_comm_hook_observed") == p.get("ddp_comm_hook", "none")
    if config.distributed.strategy == "fsdp" and gpus > 1:
        checks["FSDP wrap policy"] = metrics.get("fsdp_auto_wrap_policy") == p.get("fsdp_wrap_policy")
    if config.metadata.get("allocation_gpus"):
        try:
            placement = json.loads(metrics.get("training_rank_placement", ""))
        except (TypeError, ValueError):
            placement = []
        checks["rank placement"] = (
            isinstance(placement, list)
            and len(placement) == gpus
            and {item.get("rank") for item in placement if isinstance(item, dict)} == set(range(gpus))
            and all(item.get("gpu_uuid") not in (None, "", "unavailable") for item in placement)
            and all(
                item.get("cuda_visible_devices") == config.runtime.environment.get("CUDA_VISIBLE_DEVICES")
                for item in placement
            )
        )
    return [name for name, passed in checks.items() if not passed]


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _positive(value: Any) -> bool:
    return _finite(value) and value > 0
