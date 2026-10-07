from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.config.identity import config_sha256
from hpc_ai_perf.validation.capabilities import capability_issues


@dataclass(frozen=True)
class SweepCandidate:
    """One deterministic candidate produced by a dry-run sweep plan."""

    ordinal: int
    experiment_id: str
    config: ExperimentConfig
    axes: dict[str, Any]
    issues: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Whether the candidate passes the project's semantic capability checks."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready review record without writing a config file."""
        return {
            "ordinal": self.ordinal,
            "experiment_id": self.experiment_id,
            "valid": self.valid,
            "issues": list(self.issues),
            "axes": self.axes,
            "config_sha256": config_sha256(self.config),
            "config": self.config.to_dict(),
        }


def plan_sweep(
    base_config: ExperimentConfig,
    *,
    plan_id: str,
    axes: Mapping[str, Sequence[Any]],
) -> list[SweepCandidate]:
    """Generate a deterministic, capability-filtered candidate sweep.

    The planner is intentionally non-mutating: it returns candidate configs for
    review and dry-run output but does not write files or replace existing E###
    configurations. Each candidate receives a stable ID derived from its
    normalized configuration, so a later materialization step can be resumable.
    """
    if not plan_id or any(character.isspace() for character in plan_id):
        raise ValueError("plan_id must be non-empty and contain no whitespace")
    if not axes:
        raise ValueError("at least one sweep axis is required")
    if any(not values for values in axes.values()):
        raise ValueError("sweep axes must contain at least one value")

    axis_items = list(axes.items())
    candidates: list[SweepCandidate] = []
    value_product = itertools.product(*(values for _, values in axis_items))
    for ordinal, values in enumerate(value_product, start=1):
        axis_values = {path: value for (path, _), value in zip(axis_items, values, strict=True)}
        data = json.loads(json.dumps(base_config.to_dict()))
        for path, value in axis_values.items():
            _set_path(data, path, value)
        metadata = data.setdefault("metadata", {})
        metadata["sweep_plan_id"] = plan_id
        metadata["sweep_source_experiment_id"] = base_config.experiment_id
        metadata["sweep_axis_values"] = axis_values
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[
            :12
        ]
        data["experiment_id"] = f"{plan_id}-{digest}"
        candidate_config = ExperimentConfig.from_mapping(data)
        issues = tuple(capability_issues(candidate_config))
        candidates.append(
            SweepCandidate(
                ordinal=ordinal,
                experiment_id=candidate_config.experiment_id,
                config=candidate_config,
                axes=axis_values,
                issues=issues,
            )
        )
    return candidates


def _set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted mapping path and reject malformed sweep dimensions."""
    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("sweep axis path must not be empty")
    target: dict[str, Any] = mapping
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            raise ValueError(f"sweep axis path does not address a mapping: {path}")
        target = nested
    target[parts[-1]] = value
