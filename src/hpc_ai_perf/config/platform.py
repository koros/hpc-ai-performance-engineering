from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hpc_ai_perf.config.catalog import ConfigCatalogEntry, discover_experiment_configs
from hpc_ai_perf.config.experiment import ExperimentConfig

_EXPERIMENT_ID = re.compile(r"^E([0-9]{3})$")
_EXPERIMENT_RANGE = re.compile(r"^E([0-9]{3})-E([0-9]{3})$")
COMPARABILITY_VALUES = {"exact", "capacity_adjusted", "platform_specific"}


@dataclass(frozen=True)
class PlatformProfile:
    """Tracked hardware and scheduler policy for one execution platform."""

    platform_id: str
    platform: str
    gpu_model: str
    interconnect: str
    partition: str
    cpus_per_gpu: int
    max_gpus: int
    output_root: str
    slurm_profile: str
    unsupported_metrics: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PlatformProfile:
        required = {
            "platform_id",
            "platform",
            "gpu_model",
            "interconnect",
            "partition",
            "cpus_per_gpu",
            "max_gpus",
            "output_root",
            "slurm_profile",
        }
        missing = sorted(required - set(data))
        unknown = sorted(set(data) - required - {"schema_version", "unsupported_metrics"})
        if missing:
            raise ValueError(f"Platform profile is missing field(s): {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown platform profile field(s): {', '.join(unknown)}")
        profile = cls(
            platform_id=str(data["platform_id"]),
            platform=str(data["platform"]),
            gpu_model=str(data["gpu_model"]),
            interconnect=str(data["interconnect"]),
            partition=str(data["partition"]),
            cpus_per_gpu=int(data["cpus_per_gpu"]),
            max_gpus=int(data["max_gpus"]),
            output_root=str(data["output_root"]),
            slurm_profile=str(data["slurm_profile"]),
            unsupported_metrics=tuple(str(item) for item in data.get("unsupported_metrics", [])),
        )
        if not re.fullmatch(r"[a-z0-9_]+", profile.platform_id):
            raise ValueError("platform_id must contain only lowercase letters, numbers, and underscores")
        if profile.cpus_per_gpu < 1 or profile.max_gpus < 1:
            raise ValueError("cpus_per_gpu and max_gpus must be positive")
        return profile


@dataclass(frozen=True)
class PlatformCondition:
    """One platform-specific execution condition derived from a matrix row."""

    matrix_id: str
    platform_id: str
    wave: str
    wave_name: str
    required_trials: int
    comparability: str
    status: str = "planned"
    reason: str = ""

    @property
    def condition_id(self) -> str:
        return f"{self.matrix_id}@{self.platform_id}"


def load_platform_profile(path: str | Path) -> PlatformProfile:
    profile_path = Path(path)
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Platform profile must be a JSON object: {profile_path}")
    return PlatformProfile.from_mapping(data)


def load_platform_conditions(
    path: str | Path,
    platform_id: str,
    *,
    wave: str | None = None,
    experiment_ids: set[str] | None = None,
    include_deferred: bool = True,
) -> list[PlatformCondition]:
    """Expand tracked wave/range declarations into platform conditions."""
    matrix_path = Path(path)
    data = json.loads(matrix_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Execution matrix must be a JSON object: {matrix_path}")
    conditions: list[PlatformCondition] = []
    for wave_data in data.get("waves", []):
        if not isinstance(wave_data, dict) or str(wave_data.get("platform")) != platform_id:
            continue
        wave_id = str(wave_data["wave"])
        if wave is not None and wave_id != wave:
            continue
        comparability = str(wave_data.get("comparability", "exact"))
        if comparability not in COMPARABILITY_VALUES:
            raise ValueError(f"Unsupported comparability value: {comparability}")
        ids = _expand_experiment_selectors(wave_data.get("experiment_ids", []))
        ids -= _expand_experiment_selectors(wave_data.get("exclude_ids", []))
        for experiment_id in sorted(ids, key=_experiment_number):
            conditions.append(
                PlatformCondition(
                    matrix_id=experiment_id,
                    platform_id=platform_id,
                    wave=wave_id,
                    wave_name=str(wave_data["name"]),
                    required_trials=int(wave_data.get("required_trials", 1)),
                    comparability=comparability,
                )
            )
    if include_deferred and wave in {None, "deferred"}:
        for deferred in data.get("deferred", []):
            if not isinstance(deferred, dict) or platform_id not in deferred.get("platforms", []):
                continue
            ids = _expand_experiment_selectors(deferred.get("experiment_ids", []))
            for experiment_id in sorted(ids, key=_experiment_number):
                conditions.append(
                    PlatformCondition(
                        matrix_id=experiment_id,
                        platform_id=platform_id,
                        wave="deferred",
                        wave_name=str(deferred.get("name", "Deferred")),
                        required_trials=0,
                        comparability=str(deferred.get("comparability", "platform_specific")),
                        status="deferred",
                        reason=str(deferred["reason"]),
                    )
                )
    if experiment_ids is not None:
        conditions = [condition for condition in conditions if condition.matrix_id in experiment_ids]
    _validate_unique_conditions(conditions)
    return sorted(
        conditions,
        key=lambda condition: (_wave_sort_key(condition.wave), _experiment_number(condition.matrix_id)),
    )


def materialize_platform_configs(
    profile: PlatformProfile,
    conditions: list[PlatformCondition],
    output_dir: str | Path,
    *,
    config_root: str | Path = "configs",
    include_deferred: bool = False,
) -> list[Path]:
    """Write deterministic resolved configs without mutating source configs."""
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs(config_root)}
    output_root = Path(output_dir)
    written: list[Path] = []
    for condition in conditions:
        if condition.platform_id != profile.platform_id:
            raise ValueError(f"Condition {condition.condition_id} does not match profile {profile.platform_id}")
        if condition.status == "deferred" and not include_deferred:
            continue
        try:
            entry = entries[condition.matrix_id]
        except KeyError as exc:
            raise ValueError(f"Unknown matrix experiment in platform condition: {condition.matrix_id}") from exc
        if entry.config.hardware.gpus > profile.max_gpus:
            raise ValueError(
                f"{condition.condition_id} requests {entry.config.hardware.gpus} GPUs, "
                f"but {profile.platform_id} allows {profile.max_gpus}"
            )
        resolved = resolve_platform_config(entry, profile, condition)
        output_path = output_root / f"{condition.matrix_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(resolved.to_dict(), indent=2) + "\n", encoding="utf-8")
        written.append(output_path)
    return written


def resolve_platform_config(
    entry: ConfigCatalogEntry,
    profile: PlatformProfile,
    condition: PlatformCondition,
) -> ExperimentConfig:
    """Apply hardware/output metadata for one platform condition."""
    source_path = entry.path
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    metadata = {
        **entry.config.metadata,
        "matrix_id": condition.matrix_id,
        "condition_id": condition.condition_id,
        "execution_platform": profile.platform_id,
        "platform_wave": condition.wave,
        "platform_wave_name": condition.wave_name,
        "required_trials": condition.required_trials,
        "comparability": condition.comparability,
        "source_config_path": str(source_path),
        "source_config_sha256": source_sha256,
        "unsupported_platform_metrics": list(profile.unsupported_metrics),
    }
    if condition.status == "deferred":
        metadata["execution_status"] = "deferred"
        metadata["defer_reason"] = condition.reason
    else:
        metadata.pop("execution_status", None)
        metadata.pop("defer_reason", None)
    hardware = replace(
        entry.config.hardware,
        platform=profile.platform,
        gpu_model=profile.gpu_model,
        interconnect=profile.interconnect,
    )
    runtime = replace(entry.config.runtime, output_dir=profile.output_root)
    return replace(entry.config, hardware=hardware, runtime=runtime, metadata=metadata)


def _expand_experiment_selectors(selectors: object) -> set[str]:
    if not isinstance(selectors, list):
        raise ValueError("experiment_ids must be a list")
    expanded: set[str] = set()
    for selector_value in selectors:
        selector = str(selector_value)
        single_match = _EXPERIMENT_ID.fullmatch(selector)
        if single_match:
            expanded.add(selector)
            continue
        range_match = _EXPERIMENT_RANGE.fullmatch(selector)
        if range_match:
            start, end = (int(value) for value in range_match.groups())
            if start > end:
                raise ValueError(f"Experiment range is reversed: {selector}")
            expanded.update(f"E{index:03d}" for index in range(start, end + 1))
            continue
        raise ValueError(f"Invalid experiment selector: {selector}")
    return expanded


def _validate_unique_conditions(conditions: list[PlatformCondition]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for condition in conditions:
        if condition.condition_id in seen:
            duplicates.add(condition.condition_id)
        seen.add(condition.condition_id)
        if condition.required_trials < 0:
            raise ValueError(f"required_trials cannot be negative: {condition.condition_id}")
    if duplicates:
        raise ValueError("Duplicate platform conditions: " + ", ".join(sorted(duplicates)))


def _experiment_number(experiment_id: str) -> int:
    match = _EXPERIMENT_ID.fullmatch(experiment_id)
    if match is None:
        raise ValueError(f"Invalid matrix experiment ID: {experiment_id}")
    return int(match.group(1))


def _wave_sort_key(wave: str) -> tuple[int, str]:
    return (int(wave), wave) if wave.isdigit() else (10_000, wave)
