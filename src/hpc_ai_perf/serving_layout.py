"""Serving-layout contracts shared by validation, execution, and analysis."""

from __future__ import annotations

import re
from typing import Any

from hpc_ai_perf.config import ExperimentConfig

TP_LAYOUT_RE = re.compile(r"^tp([1-9][0-9]*)$")
REPLICA_LAYOUT = "tp2_dp2"
SUPPORTED_REQUEST_ROUTING = {"round_robin", "least_loaded"}


def declared_serving_layout(parameters: dict[str, Any]) -> str:
    """Return the canonical layout name without inventing replica semantics."""
    explicit = parameters.get("serving_layout")
    if explicit is not None:
        return str(explicit).strip().lower()

    tensor_parallel = _int_value(parameters.get("tensor_parallel_size"), default=1)
    data_parallel = _int_value(parameters.get("data_parallel_size"), default=1)
    replica_count = _int_value(parameters.get("replica_count"), default=1)
    if data_parallel == 1 and replica_count == 1:
        return f"tp{tensor_parallel}"
    return "custom"


def serving_layout_contract_requested(parameters: dict[str, Any]) -> bool:
    """Identify configs that opt into the explicit serving-layout contract."""
    return any(key in parameters for key in ("serving_layout", "replica_count", "request_routing"))


def serving_layout_issues(config: ExperimentConfig) -> list[str]:
    """Validate explicit serving layouts, including the multi-replica shape."""
    parameters = config.workload.parameters
    if config.mode != "inference" or not serving_layout_contract_requested(parameters):
        return []

    layout = declared_serving_layout(parameters)
    framework = str(parameters.get("framework") or "pytorch_eager").lower()
    tensor_parallel = _int_value(parameters.get("tensor_parallel_size"), default=1)
    data_parallel = _int_value(parameters.get("data_parallel_size"), default=1)
    replica_count = _int_value(parameters.get("replica_count"), default=1)
    request_routing = str(parameters.get("request_routing", "")).strip().lower()
    issues: list[str] = []

    if framework != "vllm":
        issues.append("explicit serving_layout currently requires framework=vllm")

    if layout == REPLICA_LAYOUT:
        if tensor_parallel != 2:
            issues.append("serving_layout=tp2_dp2 requires tensor_parallel_size=2")
        if data_parallel != 2:
            issues.append("serving_layout=tp2_dp2 requires data_parallel_size=2")
        if replica_count != 2:
            issues.append("serving_layout=tp2_dp2 requires replica_count=2")
        if config.hardware.gpus != 4:
            issues.append("serving_layout=tp2_dp2 requires hardware.gpus=4")
        if request_routing not in SUPPORTED_REQUEST_ROUTING:
            issues.append("serving_layout=tp2_dp2 requires request_routing=round_robin or request_routing=least_loaded")
        return issues

    match = TP_LAYOUT_RE.fullmatch(layout)
    if match is None:
        issues.append(f"unknown serving_layout={layout!r}; use tpN or {REPLICA_LAYOUT}")
        return issues

    expected_tensor_parallel = int(match.group(1))
    if tensor_parallel != expected_tensor_parallel:
        issues.append(f"serving_layout={layout} requires tensor_parallel_size={expected_tensor_parallel}")
    if data_parallel != 1:
        issues.append(f"serving_layout={layout} requires data_parallel_size=1")
    if replica_count != 1:
        issues.append(f"serving_layout={layout} requires replica_count=1")
    if config.hardware.gpus != expected_tensor_parallel:
        issues.append(f"serving_layout={layout} requires hardware.gpus={expected_tensor_parallel}")
    if request_routing and request_routing != "single_process":
        issues.append(f"serving_layout={layout} does not support request_routing={request_routing!r}")
    return issues


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
