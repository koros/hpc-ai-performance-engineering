"""Dry-run plans for explicit multi-engine serving layouts."""

from __future__ import annotations

import os
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.serving_layout import declared_serving_layout
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.workloads.vllm_router import build_gpu_groups


def plan_serving_layout(
    config: ExperimentConfig,
    *,
    visible_devices: str | None = None,
) -> dict[str, Any]:
    """Resolve a serving layout without importing or launching a model engine."""
    parameters = config.workload.parameters
    layout = declared_serving_layout(parameters)
    tensor_parallel_size = _positive_int(parameters.get("tensor_parallel_size"), default=1)
    data_parallel_size = _positive_int(parameters.get("data_parallel_size"), default=1)
    replica_count = _positive_int(parameters.get("replica_count"), default=1)
    issues = list(capability_issues(config))
    gpu_groups: tuple[tuple[str, ...], ...] = ()
    try:
        gpu_groups = build_gpu_groups(
            gpu_count=config.hardware.gpus,
            tensor_parallel_size=tensor_parallel_size,
            replica_count=replica_count,
            visible_devices=visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    except ValueError as exc:
        issues.append(str(exc))

    return {
        "experiment_id": config.experiment_id,
        "workload": config.workload.name,
        "framework": parameters.get("framework", "pytorch_eager"),
        "serving_layout": layout,
        "request_routing": parameters.get("request_routing", "single_process"),
        "hardware_gpus": config.hardware.gpus,
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "replica_count": replica_count,
        "concurrent_requests": parameters.get("concurrent_requests", 1),
        "gpu_groups": [list(group) for group in gpu_groups],
        "capability_issues": issues,
        "valid": not issues,
    }


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
