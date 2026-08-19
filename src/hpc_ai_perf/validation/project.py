from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.config.traceability import ANALYSIS_TASKS, RESEARCH_QUESTIONS
from hpc_ai_perf.validation.capabilities import capability_issues, unknown_parameter_issues

CONFIG_MODES = {"training", "inference", "profiling"}
PRECISIONS = {"fp32", "fp16", "bf16", "fp8", "int8", "int4"}
RESULT_STATUSES = {"completed", "failed"}
STRATEGIES = {"none", "ddp", "fsdp", "deepspeed_zero1", "deepspeed_zero2", "deepspeed_zero3"}
THROUGHPUT_UNITS = {
    "generated_tokens_per_second",
    "input_tokens_per_second",
    "synthetic_tokens_per_second",
    "training_tokens_per_second",
}

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
    """One validation problem found in a config or raw result file."""

    path: Path
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-friendly representation for CLI automation."""
        return {"path": str(self.path), "severity": self.severity, "message": self.message}


def validate_project(
    config_root: str | Path = "configs",
    result_dir: str | Path = "results/raw",
    *,
    include_results: bool = True,
    research_ready: bool = False,
) -> list[ValidationIssue]:
    """Validate configs and, optionally, raw result records."""
    issues = validate_experiment_configs(config_root, research_ready=research_ready)
    if include_results:
        issues.extend(validate_result_records(result_dir))
    return issues


def validate_experiment_configs(
    config_root: str | Path = "configs", *, research_ready: bool = False
) -> list[ValidationIssue]:
    """Validate every JSON experiment config under a root directory.

    This pass checks required fields, enum-like values, non-negative hardware
    counts, runtime settings, and matrix-specific metadata. It collects all
    issues instead of failing fast so a user can repair many config files in one
    editing pass.
    """
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
        issues.extend(ValidationIssue(config_path, message) for message in unknown_parameter_issues(config))

        matrix_id = str(config.metadata.get("matrix_id", config.experiment_id))
        if _is_matrix_experiment_id(config.experiment_id) and matrix_id != config.experiment_id:
            issues.append(ValidationIssue(config_path, "metadata.matrix_id must match experiment_id"))
        if _is_matrix_experiment_id(config.experiment_id):
            if not config.metadata.get("phase"):
                issues.append(ValidationIssue(config_path, "metadata.phase is required for matrix configs"))
            if not config.metadata.get("priority"):
                issues.append(ValidationIssue(config_path, "metadata.priority is required for matrix configs"))
            research_questions = config.metadata.get("research_questions")
            if not _is_known_identifier_list(research_questions, RESEARCH_QUESTIONS):
                issues.append(
                    ValidationIssue(config_path, "metadata.research_questions must contain valid RQ1-RQ7 identifiers")
                )
            analysis_tasks = config.metadata.get("analysis_tasks")
            if not _is_known_identifier_list(analysis_tasks, ANALYSIS_TASKS):
                issues.append(
                    ValidationIssue(config_path, "metadata.analysis_tasks must contain valid A01-A08 identifiers")
                )
            if research_ready:
                issues.extend(
                    ValidationIssue(config_path, message, severity="error") for message in capability_issues(config)
                )

    return issues


def validate_result_records(result_dir: str | Path = "results/raw") -> list[ValidationIssue]:
    """Validate every raw result JSON record under a result directory."""
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
    """Format validation issues for terminal output."""
    if not issues:
        return "Validation passed.\n"

    lines = ["Validation failed:", ""]
    for issue in issues:
        lines.append(f"- {issue.severity.upper()} {issue.path}: {issue.message}")
    return "\n".join(lines) + "\n"


def _validate_result_mapping(path: Path, data: dict[str, Any]) -> list[ValidationIssue]:
    """Validate the schema and scalar metric shape of one parsed result record."""
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
    if "repetition_number" in data and not _is_positive_int(data["repetition_number"]):
        issues.append(ValidationIssue(path, "repetition_number must be a positive integer"))
    for field in ["warmup_iterations", "measured_iterations"]:
        if field in data and data[field] is not None and not _is_non_negative_int(data[field]):
            issues.append(ValidationIssue(path, f"{field} must be null or a non-negative integer"))

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

    experiment_id = str(data.get("experiment_id", ""))
    if _is_matrix_experiment_id(experiment_id) and data.get("status") == "completed":
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        environment = data.get("environment") if isinstance(data.get("environment"), dict) else {}
        config = data.get("config") if isinstance(data.get("config"), dict) else {}
        workload = config.get("workload") if isinstance(config.get("workload"), dict) else {}
        if not data.get("run_id"):
            issues.append(ValidationIssue(path, "completed matrix result must include run_id"))
        for field in [
            "repetition_number",
            "warmup_iterations",
            "measured_iterations",
            "git_commit",
            "git_dirty",
            "config_sha256",
            "command",
        ]:
            if field not in data or data[field] is None:
                issues.append(ValidationIssue(path, f"completed matrix result must include {field}"))
        if data.get("git_dirty") is not False:
            issues.append(ValidationIssue(path, "completed matrix evidence must come from a clean git worktree"))
        recorded_hash = data.get("config_sha256")
        if isinstance(recorded_hash, str) and config:
            expected_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if recorded_hash != expected_hash:
                issues.append(ValidationIssue(path, "config_sha256 does not match the embedded config"))
        if int(data.get("gpus", 0) or 0) > 0 and not environment.get("observed_gpu_model"):
            issues.append(ValidationIssue(path, "completed GPU result must record environment.observed_gpu_model"))
        if workload.get("name") != "environment_validation":
            hardware = config.get("hardware") if isinstance(config.get("hardware"), dict) else {}
            requested_model = str(hardware.get("gpu_model", "")).lower()
            observed_model = str(environment.get("observed_gpu_model", "")).lower()
            expected_token = "h100" if "h100" in requested_model else "l40s" if "l40s" in requested_model else ""
            if expected_token and observed_model and expected_token not in observed_model:
                gpu_model = environment.get("observed_gpu_model")
                issues.append(
                    ValidationIssue(
                        path,
                        f"observed GPU model does not match requested hardware: {gpu_model}",
                    )
                )
            if not data.get("model_revision"):
                issues.append(ValidationIssue(path, "completed transformer result must include model_revision"))
            if not data.get("dataset_revision"):
                issues.append(ValidationIssue(path, "completed transformer result must include dataset_revision"))
            if data.get("throughput_unit") not in THROUGHPUT_UNITS:
                issues.append(
                    ValidationIssue(path, "completed matrix result must include a recognized throughput_unit")
                )
            for metric in ["measured_seconds", "gpu_hours", "total_tokens"]:
                if not _is_non_negative_number(metrics.get(metric)):
                    issues.append(ValidationIssue(path, f"completed matrix result must include metric: {metric}"))

    return issues


def _is_matrix_experiment_id(experiment_id: str) -> bool:
    """Return whether an ID is part of the canonical dissertation matrix."""
    return len(experiment_id) == 4 and experiment_id.startswith("E") and experiment_id[1:].isdigit()


def _is_non_negative_int(value: object) -> bool:
    """Return whether a value is an integer greater than or equal to zero."""
    return isinstance(value, int) and value >= 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and value >= 1


def _is_non_negative_number(value: object) -> bool:
    """Return whether a value is a numeric value greater than or equal to zero."""
    return isinstance(value, int | float) and value >= 0


def _is_optional_non_negative_number(value: object) -> bool:
    """Return whether a value is null or a non-negative number."""
    return value is None or _is_non_negative_number(value)


def _is_optional_percentage(value: object) -> bool:
    """Return whether a value is null or a percentage in the inclusive range 0-100."""
    return value is None or (_is_non_negative_number(value) and value <= 100)


def _is_scalar_mapping(value: object) -> bool:
    """Return whether a mapping contains only scalar JSON-compatible values."""
    if not isinstance(value, dict):
        return False
    return all(item is None or isinstance(item, int | float | str | bool) for item in value.values())


def _is_known_identifier_list(value: object, allowed: set[str]) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item in allowed for item in value)
