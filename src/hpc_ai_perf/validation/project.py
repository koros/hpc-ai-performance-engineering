from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import load_experiment_config

CONFIG_MODES = {"training", "inference", "profiling"}
PRECISIONS = {"fp32", "fp16", "bf16", "fp8", "int8", "int4"}
RESULT_STATUSES = {"completed", "failed"}
STRATEGIES = {"none", "ddp", "fsdp", "deepspeed_zero1", "deepspeed_zero2", "deepspeed_zero3"}

RESULT_REQUIRED_FIELDS = [
    "schema_version",
    "experiment_id",
    "experiment",
    "workload",
    "model",
    "mode",
    "gpus",
    "precision",
    "status",
    "started_at",
    "completed_at",
    "runtime_seconds",
    "metrics",
    "environment",
    "config",
]


@dataclass(frozen=True)
class ValidationIssue:
    path: Path
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "severity": self.severity, "message": self.message}


def validate_project(
    config_root: str | Path = "configs",
    result_dir: str | Path = "results/raw",
    *,
    include_results: bool = True,
) -> list[ValidationIssue]:
    issues = validate_experiment_configs(config_root)
    if include_results:
        issues.extend(validate_result_records(result_dir))
    return issues


def validate_experiment_configs(config_root: str | Path = "configs") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for config_path in sorted(Path(config_root).rglob("*.json")):
        try:
            config = load_experiment_config(config_path)
        except Exception as exc:
            issues.append(ValidationIssue(config_path, f"could not load config: {exc}"))
            continue

        if not config.experiment_id:
            issues.append(ValidationIssue(config_path, "experiment_id is required"))
        if not config.experiment:
            issues.append(ValidationIssue(config_path, "experiment is required"))
        if config.mode not in CONFIG_MODES:
            issues.append(ValidationIssue(config_path, f"unsupported mode: {config.mode}"))
        if not config.workload.name:
            issues.append(ValidationIssue(config_path, "workload.name is required"))
        if not config.workload.model:
            issues.append(ValidationIssue(config_path, "workload.model is required"))
        if not isinstance(config.hardware.gpus, int) or config.hardware.gpus < 0:
            issues.append(ValidationIssue(config_path, "hardware.gpus must be a non-negative integer"))
        if config.distributed.strategy not in STRATEGIES:
            issues.append(
                ValidationIssue(config_path, f"unsupported distributed.strategy: {config.distributed.strategy}")
            )
        if config.distributed.nodes < 1:
            issues.append(ValidationIssue(config_path, "distributed.nodes must be at least 1"))
        if config.distributed.gpus_per_node < 0:
            issues.append(ValidationIssue(config_path, "distributed.gpus_per_node must be non-negative"))
        if config.runtime.precision not in PRECISIONS:
            issues.append(ValidationIssue(config_path, f"unsupported runtime.precision: {config.runtime.precision}"))
        if config.runtime.max_steps is not None and config.runtime.max_steps < 1:
            issues.append(ValidationIssue(config_path, "runtime.max_steps must be positive when provided"))
        if not config.runtime.output_dir:
            issues.append(ValidationIssue(config_path, "runtime.output_dir is required"))

        matrix_id = str(config.metadata.get("matrix_id", config.experiment_id))
        if _is_matrix_experiment_id(config.experiment_id) and matrix_id != config.experiment_id:
            issues.append(ValidationIssue(config_path, "metadata.matrix_id must match experiment_id"))
        if _is_matrix_experiment_id(config.experiment_id):
            if not config.metadata.get("phase"):
                issues.append(ValidationIssue(config_path, "metadata.phase is required for matrix configs"))
            if not config.metadata.get("priority"):
                issues.append(ValidationIssue(config_path, "metadata.priority is required for matrix configs"))

    return issues


def validate_result_records(result_dir: str | Path = "results/raw") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for result_path in sorted(Path(result_dir).rglob("*.json")):
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(ValidationIssue(result_path, f"invalid JSON: {exc}"))
            continue

        if not isinstance(data, dict):
            issues.append(ValidationIssue(result_path, "result record must be a JSON object"))
            continue

        issues.extend(_validate_result_mapping(result_path, data))

    return issues


def format_validation_issues(issues: list[ValidationIssue]) -> str:
    if not issues:
        return "Validation passed.\n"

    lines = ["Validation failed:", ""]
    for issue in issues:
        lines.append(f"- {issue.severity.upper()} {issue.path}: {issue.message}")
    return "\n".join(lines) + "\n"


def _validate_result_mapping(path: Path, data: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in RESULT_REQUIRED_FIELDS:
        if field not in data:
            issues.append(ValidationIssue(path, f"missing required result field: {field}"))

    if "mode" in data and data["mode"] not in CONFIG_MODES:
        issues.append(ValidationIssue(path, f"unsupported mode: {data['mode']}"))
    if "precision" in data and data["precision"] not in PRECISIONS:
        issues.append(ValidationIssue(path, f"unsupported precision: {data['precision']}"))
    if "status" in data and data["status"] not in RESULT_STATUSES:
        issues.append(ValidationIssue(path, f"unsupported status: {data['status']}"))
    if "gpus" in data and not _is_non_negative_int(data["gpus"]):
        issues.append(ValidationIssue(path, "gpus must be a non-negative integer"))
    if "runtime_seconds" in data and not _is_non_negative_number(data["runtime_seconds"]):
        issues.append(ValidationIssue(path, "runtime_seconds must be a non-negative number"))

    for field in [
        "throughput_tokens_sec",
        "throughput_samples_sec",
        "memory_used_gb",
        "latency_p50_ms",
        "latency_p95_ms",
    ]:
        if field in data and not _is_optional_non_negative_number(data[field]):
            issues.append(ValidationIssue(path, f"{field} must be null or a non-negative number"))

    if "gpu_utilization" in data and not _is_optional_percentage(data["gpu_utilization"]):
        issues.append(ValidationIssue(path, "gpu_utilization must be null or a percentage from 0 to 100"))
    if "metrics" in data and not _is_scalar_mapping(data["metrics"]):
        issues.append(ValidationIssue(path, "metrics must be an object containing scalar values"))
    if "environment" in data and not _is_scalar_mapping(data["environment"]):
        issues.append(ValidationIssue(path, "environment must be an object containing scalar values"))
    if "config" in data and not isinstance(data["config"], dict):
        issues.append(ValidationIssue(path, "config must be an object"))

    return issues


def _is_matrix_experiment_id(experiment_id: str) -> bool:
    return len(experiment_id) == 4 and experiment_id.startswith("E") and experiment_id[1:].isdigit()


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and value >= 0


def _is_non_negative_number(value: object) -> bool:
    return isinstance(value, int | float) and value >= 0


def _is_optional_non_negative_number(value: object) -> bool:
    return value is None or _is_non_negative_number(value)


def _is_optional_percentage(value: object) -> bool:
    return value is None or (_is_non_negative_number(value) and value <= 100)


def _is_scalar_mapping(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(item is None or isinstance(item, int | float | str | bool) for item in value.values())
