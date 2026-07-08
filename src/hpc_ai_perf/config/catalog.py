from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpc_ai_perf.config.experiment import ExperimentConfig, load_experiment_config


@dataclass(frozen=True)
class ConfigCatalogEntry:
    path: Path
    config: ExperimentConfig

    @property
    def phase(self) -> str:
        return str(self.config.metadata.get("phase", ""))

    @property
    def priority(self) -> str:
        return str(self.config.metadata.get("priority", ""))

    @property
    def matrix_id(self) -> str:
        return str(self.config.metadata.get("matrix_id", self.config.experiment_id))

    def to_summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.config.experiment_id,
            "matrix_id": self.matrix_id,
            "phase": self.phase,
            "priority": self.priority,
            "mode": self.config.mode,
            "workload": self.config.workload.name,
            "strategy": self.config.distributed.strategy,
            "gpus": self.config.hardware.gpus,
            "precision": self.config.runtime.precision,
            "path": str(self.path),
        }


def discover_experiment_configs(
    config_root: str | Path = "configs",
    *,
    matrix_only: bool = True,
) -> list[ConfigCatalogEntry]:
    root = Path(config_root)
    entries: list[ConfigCatalogEntry] = []

    for config_path in sorted(root.rglob("*.json")):
        config = load_experiment_config(config_path)
        if matrix_only and not _is_matrix_experiment_id(config.experiment_id):
            continue
        entries.append(ConfigCatalogEntry(path=config_path, config=config))

    return sorted(entries, key=_catalog_sort_key)


def filter_config_entries(
    entries: list[ConfigCatalogEntry],
    *,
    phase: str | None = None,
    priority: str | None = None,
    mode: str | None = None,
    workload: str | None = None,
    experiment_id: str | None = None,
) -> list[ConfigCatalogEntry]:
    filtered = entries
    if phase:
        filtered = [entry for entry in filtered if entry.phase.lower() == phase.lower()]
    if priority:
        filtered = [entry for entry in filtered if entry.priority.lower() == priority.lower()]
    if mode:
        filtered = [entry for entry in filtered if entry.config.mode == mode]
    if workload:
        filtered = [entry for entry in filtered if entry.config.workload.name == workload]
    if experiment_id:
        filtered = [entry for entry in filtered if entry.config.experiment_id == experiment_id]
    return filtered


def entries_to_json(entries: list[ConfigCatalogEntry]) -> str:
    return json.dumps([entry.to_summary() for entry in entries], indent=2) + "\n"


def entries_to_table(entries: list[ConfigCatalogEntry]) -> str:
    headers = ["ID", "Phase", "Priority", "Mode", "Workload", "GPUs", "Precision", "Path"]
    rows = [
        [
            entry.config.experiment_id,
            entry.phase,
            entry.priority,
            entry.config.mode,
            entry.config.workload.name,
            str(entry.config.hardware.gpus),
            entry.config.runtime.precision,
            str(entry.path),
        ]
        for entry in entries
    ]
    return _format_table(headers, rows)


def _is_matrix_experiment_id(experiment_id: str) -> bool:
    return len(experiment_id) == 4 and experiment_id.startswith("E") and experiment_id[1:].isdigit()


def _catalog_sort_key(entry: ConfigCatalogEntry) -> tuple[int, str]:
    if _is_matrix_experiment_id(entry.config.experiment_id):
        return int(entry.config.experiment_id[1:]), str(entry.path)
    return 10_000, str(entry.path)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "No configs found.\n"

    widths = [max(len(headers[column]), *(len(row[column]) for row in rows)) for column in range(len(headers))]
    header_line = "  ".join(headers[column].ljust(widths[column]) for column in range(len(headers)))
    separator = "  ".join("-" * widths[column] for column in range(len(headers)))
    body = ["  ".join(row[column].ljust(widths[column]) for column in range(len(headers))) for row in rows]
    return "\n".join([header_line, separator, *body]) + "\n"
