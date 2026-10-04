"""Collected, platform-specific acceptance gate for the TinyLlama batch crossover."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.validation import validate_result_file
from hpc_ai_perf.validation.qwen7b_training import canary_contract_sha256
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256

CANARY_IDS = {"E314", "E315", "E316", "E317"}


def verify_scaling_canaries(
    result_root: str | Path,
    platform: str,
    *,
    execution_matrix: str | Path = "platforms/execution_matrix.json",
) -> dict[str, dict[str, Any]]:
    """Require four current canaries and the same global B=8 inputs at 1/2/4 GPUs."""
    if platform not in {"l40s", "h100"}:
        raise ValueError("Scaling acceptance platform must be l40s or h100")
    root = Path(result_root)
    profile = load_platform_profile(f"platforms/{platform}.json")
    conditions = {
        condition.matrix_id: condition
        for condition in load_platform_conditions(execution_matrix, platform, wave="scaling-batch-canary")
    }
    if set(conditions) != CANARY_IDS:
        raise ValueError("Scaling acceptance matrix does not contain the four required canaries")
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs("configs/scaling_studies")}
    checksums = _manifest_entries(root / "SHA256SUMS")
    issues: list[str] = []
    accepted: dict[str, dict[str, Any]] = {}
    for eid in sorted(CANARY_IDS):
        condition = conditions[eid]
        config = resolve_platform_config(entries[eid], profile, condition)
        expected_hash = canary_contract_sha256(config)
        candidates: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted((root / eid).glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                issues.append(f"{eid}: unreadable collected record: {path.name}")
                continue
            if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
                issues.append(f"{eid}: invalid collected record: {path.name}")
                continue
            if canary_contract_sha256(record["config"]) != expected_hash:
                continue
            relpath = path.relative_to(root).as_posix()
            if checksums.get(relpath) != hashlib.sha256(path.read_bytes()).hexdigest():
                issues.append(f"{eid}: collected evidence SHA256SUMS mismatch: {path.name}")
                continue
            validation = validate_result_file(path)
            if validation:
                issues.extend(f"{eid}: invalid result: {issue.message}" for issue in validation)
                continue
            if record.get("status") == "completed" and record.get("error") is None:
                candidates.append((path, record))
        if not candidates:
            issues.append(f"{eid}: no completed, collected record matches the current canary contract")
            continue
        # A new valid retry can replace a prior failed or incomplete diagnostic,
        # but the immutable older record remains in the evidence tree.
        path, record = max(candidates, key=lambda pair: pair[1].get("completed_at", ""))
        metrics = record.get("metrics", {})
        environment = record.get("environment", {})
        p = config.workload.parameters
        updates = p["warmup_steps"] + config.runtime.max_steps
        gpus = config.hardware.gpus
        placement = _placement_ranks(metrics.get("training_rank_placement"))
        checks = {
            "platform and condition": record.get("platform_id") == platform
            and record.get("condition_id") == condition.condition_id,
            "clean source": record.get("git_dirty") is False and environment.get("git_dirty") is False,
            "GPU model": platform in str(environment.get("observed_gpu_model", "")).lower(),
            "world size": record.get("gpus") == gpus and metrics.get("world_size") == gpus,
            "runtime code": metrics.get("scaling_runtime_sha256") == scaling_runtime_sha256(),
            "input protocol": metrics.get("dataset_execution") == p["input_protocol"],
            "input digest": _hex_digest(metrics.get("global_input_sha256")),
            "optimizer updates": metrics.get("optimizer_step_calls") == updates,
            "backward calls": metrics.get("backward_calls") == updates * p["gradient_accumulation_steps"],
            "input consumption": metrics.get("input_micro_batches_consumed")
            == updates * p["gradient_accumulation_steps"],
            "batch accounting": metrics.get("effective_global_batch_size") == p["global_batch_size"]
            and metrics.get("batch_size_per_gpu") == p["per_gpu_batch_size"]
            and metrics.get("total_samples") == config.runtime.max_steps * p["global_batch_size"],
            "sequence length": metrics.get("sequence_length") == p["sequence_length"],
            "BF16 compute and FP32 parameters": metrics.get("compute_dtype_observed") == "bf16"
            and metrics.get("parameter_dtype_observed") == "torch.float32",
            "DDP strategy": metrics.get("strategy_observed") == "ddp",
            "SDPA without cache": metrics.get("attention_backend_observed") == "sdpa"
            and metrics.get("training_use_cache") is False,
            "loss": _finite(metrics.get("initial_training_loss")) and _finite(metrics.get("loss")),
            "throughput": _positive(metrics.get("throughput_tokens_sec"))
            and _positive(metrics.get("measured_seconds")),
            "memory": _positive(metrics.get("memory_used_gb")),
            "rank placement": len(placement) == gpus and {rank.get("rank") for rank in placement} == set(range(gpus)),
            "GPU telemetry": metrics.get("nvidia_smi.profiled_gpu_count") == gpus,
            "measured energy": metrics.get("nvidia_smi.measurement_status") == "sufficient"
            and _positive(metrics.get("nvidia_smi.energy_joules_measured_region")),
        }
        issues.extend(f"{eid}: {name} failed" for name, passed in checks.items() if not passed)
        accepted[eid] = {
            "result": str(path),
            "contract_sha256": expected_hash,
            "gpus": gpus,
            "global_batch": p["global_batch_size"],
            "input_sha256": metrics.get("global_input_sha256"),
            "throughput_tokens_sec": metrics.get("throughput_tokens_sec"),
        }
    b8_hashes = {accepted[eid]["input_sha256"] for eid in ("E314", "E315", "E316") if eid in accepted}
    if len(b8_hashes) != 1:
        issues.append("E314/E315/E316: global batch 8 input digests differ across GPU counts")
    if issues:
        raise ValueError("Scaling acceptance failed:\n- " + "\n- ".join(issues))
    return accepted


def _manifest_entries(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Scaling acceptance requires collected SHA256SUMS: {path}")
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed collected SHA256SUMS: {path}") from exc
        if not _hex_digest(digest) or relative in entries:
            raise ValueError(f"Malformed or duplicate collected SHA256SUMS entry: {relative}")
        entries[relative] = digest
    return entries


def _placement_ranks(value: Any) -> list[dict[str, Any]]:
    try:
        ranks = json.loads(value)
    except (TypeError, ValueError):
        return []
    return ranks if isinstance(ranks, list) and all(isinstance(rank, dict) for rank in ranks) else []


def _hex_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def _positive(value: Any) -> bool:
    return _finite(value) and value > 0
