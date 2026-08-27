from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MATRIX_PLATFORMS = {"l40s", "h100"}
COMPARABILITY_VALUES = {"exact", "capacity_adjusted", "platform_specific"}
_MATRIX_ID = re.compile(r"^E[0-9]{3}$")
_CONDITION_ID = re.compile(r"^(E[0-9]{3})@(l40s|h100)$")


@dataclass(frozen=True)
class MatrixResultIdentity:
    """Resolved identity used to prevent cross-platform evidence pooling."""

    experiment_id: str
    condition_id: str
    platform_id: str
    comparability: str
    explicit_contract: bool


def resolve_matrix_result_identity(record: dict[str, Any]) -> MatrixResultIdentity:
    """Resolve one matrix record or reject ambiguous/conflicting platform signals."""
    experiment_id = str(record.get("experiment_id", "")).strip()
    if not _MATRIX_ID.fullmatch(experiment_id):
        raise ValueError(f"not a canonical matrix experiment ID: {experiment_id or '<missing>'}")

    config = _mapping(record.get("config"))
    metadata = _mapping(config.get("metadata"))
    hardware = _mapping(config.get("hardware"))
    environment = _mapping(record.get("environment"))
    explicit_contract = _uses_explicit_contract(record)

    platform_signals: dict[str, str] = {}
    _add_explicit_platform_signal(platform_signals, "platform_id", record.get("platform_id"))
    _add_explicit_platform_signal(
        platform_signals,
        "config.metadata.execution_platform",
        metadata.get("execution_platform"),
    )
    for label, value in [
        ("config.hardware.platform", hardware.get("platform")),
        ("config.hardware.gpu_model", hardware.get("gpu_model")),
        ("environment.observed_gpu_model", environment.get("observed_gpu_model")),
        ("environment.requested_gpu_model", environment.get("requested_gpu_model")),
        ("environment.requested_platform", environment.get("requested_platform")),
    ]:
        token = _platform_token(value)
        if token:
            platform_signals[label] = token

    condition_values = {
        "condition_id": record.get("condition_id"),
        "config.metadata.condition_id": metadata.get("condition_id"),
    }
    parsed_conditions: dict[str, tuple[str, str]] = {}
    for label, value in condition_values.items():
        if value is None or str(value).strip() == "":
            continue
        match = _CONDITION_ID.fullmatch(str(value).strip())
        if match is None:
            raise ValueError(f"{label} must use E###@l40s or E###@h100: {value}")
        parsed_conditions[label] = (match.group(1), match.group(2))
        platform_signals[label] = match.group(2)

    observed_platforms = set(platform_signals.values())
    if len(observed_platforms) != 1:
        details = ", ".join(f"{label}={value}" for label, value in sorted(platform_signals.items())) or "none"
        raise ValueError(f"matrix evidence must identify exactly one L40S/H100 platform; observed {details}")
    platform_id = next(iter(observed_platforms))
    expected_condition_id = f"{experiment_id}@{platform_id}"
    for label, (condition_experiment, condition_platform) in parsed_conditions.items():
        if condition_experiment != experiment_id or condition_platform != platform_id:
            raise ValueError(f"{label} conflicts with resolved identity {expected_condition_id}")

    comparability_values: dict[str, str] = {}
    for label, value in [
        ("comparability", record.get("comparability")),
        ("config.metadata.comparability", metadata.get("comparability")),
    ]:
        if value is None or str(value).strip() == "":
            continue
        normalized = str(value).strip()
        if normalized not in COMPARABILITY_VALUES:
            raise ValueError(f"{label} has unsupported value: {normalized}")
        comparability_values[label] = normalized
    if len(set(comparability_values.values())) > 1:
        details = ", ".join(f"{label}={value}" for label, value in sorted(comparability_values.items()))
        raise ValueError(f"comparability fields conflict: {details}")
    if explicit_contract and not comparability_values:
        raise ValueError("platform-aware matrix evidence must include comparability")
    comparability = next(iter(comparability_values.values()), "platform_specific")

    if explicit_contract:
        for field in ["condition_id", "platform_id", "comparability"]:
            if record.get(field) is None or str(record.get(field)).strip() == "":
                raise ValueError(f"platform-aware matrix evidence must include top-level {field}")

    return MatrixResultIdentity(
        experiment_id=experiment_id,
        condition_id=expected_condition_id,
        platform_id=platform_id,
        comparability=comparability,
        explicit_contract=explicit_contract,
    )


def is_matrix_experiment_id(experiment_id: object) -> bool:
    return bool(_MATRIX_ID.fullmatch(str(experiment_id or "")))


def _uses_explicit_contract(record: dict[str, Any]) -> bool:
    schema_version = str(record.get("schema_version", ""))
    return schema_version.startswith("1.3.") or any(
        field in record for field in ["condition_id", "platform_id", "comparability"]
    )


def _add_explicit_platform_signal(signals: dict[str, str], label: str, value: object) -> None:
    if value is None or str(value).strip() == "":
        return
    token = _platform_token(value)
    if token is None:
        raise ValueError(f"{label} must identify l40s or h100: {value}")
    signals[label] = token


def _platform_token(value: object) -> str | None:
    text = str(value or "").lower()
    matches = {platform_id for platform_id, token in [("l40s", "l40"), ("h100", "h100")] if token in text}
    if len(matches) > 1:
        raise ValueError(f"one platform field contains conflicting L40S/H100 labels: {value}")
    return next(iter(matches), None)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
