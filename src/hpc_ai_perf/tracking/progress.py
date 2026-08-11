from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import ConfigCatalogEntry, discover_experiment_configs
from hpc_ai_perf.results import load_result_records

TRACKER_COLUMNS = [
    "experiment_id",
    "status",
    "phase",
    "workload",
    "mode",
    "strategy",
    "gpus",
    "precision",
    "priority",
    "gpu_hours_estimate",
    "result_file",
    "completed_at",
    "runtime_seconds",
    "notes",
]

RESULT_STATUS = {
    "completed": "Completed",
    "failed": "Failed",
}


def track_experiment_progress(
    config_root: str | Path = "configs",
    result_dir: str | Path = "results/raw",
    output_csv: str | Path = "results/processed/experiment_tracker.csv",
    output_markdown: str | Path = "results/processed/experiment_tracker.md",
    profile: str = "matrix",
) -> tuple[Path, Path]:
    entries = _discover_profile_entries(config_root, profile)
    result_index = _latest_result_by_experiment_id(result_dir, profile)
    rows = build_progress_rows(entries, result_index)

    csv_path = write_tracker_csv(rows, output_csv)
    markdown_path = write_tracker_markdown(rows, output_markdown)
    return csv_path, markdown_path


def build_progress_rows(
    entries: list[ConfigCatalogEntry], result_index: dict[str, tuple[Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        config = entry.config
        result_path, result = result_index.get(config.experiment_id, (None, {}))
        status = RESULT_STATUS.get(str(result.get("status", "")).lower(), "Configured")
        error = str(result.get("error") or "")
        notes = error if status == "Failed" else ""

        rows.append(
            {
                "experiment_id": config.experiment_id,
                "status": status,
                "phase": entry.phase,
                "workload": config.workload.name,
                "mode": config.mode,
                "strategy": config.distributed.strategy,
                "gpus": config.hardware.gpus,
                "precision": config.runtime.precision,
                "priority": entry.priority,
                "gpu_hours_estimate": config.metadata.get("gpu_hours_estimate", ""),
                "result_file": str(result_path) if result_path else f"results/raw/{config.experiment_id}.json",
                "completed_at": result.get("completed_at", ""),
                "runtime_seconds": result.get("runtime_seconds", ""),
                "notes": notes,
            }
        )

    return rows


def write_tracker_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=TRACKER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in TRACKER_COLUMNS})

    return csv_path


def write_tracker_markdown(rows: list[dict[str, Any]], output_markdown: str | Path) -> Path:
    markdown_path = Path(output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _status_summary(rows)
    lines = [
        "# Generated Experiment Tracker",
        "",
        "This file is generated from `configs/` and `results/raw/`.",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "| --- | ---: |",
        *[f"| {status} | {count} |" for status, count in sorted(summary.items())],
        "",
        "## Experiments",
        "",
        "| ID | Status | Phase | Workload | Mode | Strategy | GPUs | Precision | Priority | GPUh | Result file |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {experiment_id} | {status} | {phase} | {workload} | {mode} | {strategy} | {gpus} | "
            "{precision} | {priority} | {gpu_hours_estimate} | {result_file} |".format(
                **{column: _markdown_value(row.get(column)) for column in TRACKER_COLUMNS}
            )
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def _latest_result_by_experiment_id(result_dir: str | Path, profile: str) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for result_path, result in load_result_records(result_dir):
        experiment_id = str(result.get("experiment_id", ""))
        if not _result_matches_profile(experiment_id, profile):
            continue
        current = indexed.get(experiment_id)
        if current is None or _result_sort_key(result_path, result) > _result_sort_key(*current):
            indexed[experiment_id] = (result_path, result)
    return indexed


def _discover_profile_entries(config_root: str | Path, profile: str) -> list[ConfigCatalogEntry]:
    if profile == "matrix":
        return discover_experiment_configs(config_root)

    # Dev/local trackers must include non-E IDs, while the default matrix tracker must not.
    entries = discover_experiment_configs(config_root, matrix_only=False)
    if profile == "all":
        return entries

    return [entry for entry in entries if str(entry.config.metadata.get("profile", "")).lower() == profile]


def _result_matches_profile(experiment_id: str, profile: str) -> bool:
    if profile == "matrix":
        return _is_matrix_experiment_id(experiment_id)
    if profile == "all":
        return True
    # Local profiles use readable prefixes such as dev_ instead of matrix IDs.
    return experiment_id.startswith(f"{profile}_")


def _is_matrix_experiment_id(experiment_id: str) -> bool:
    return len(experiment_id) == 4 and experiment_id.startswith("E") and experiment_id[1:].isdigit()


def _result_sort_key(result_path: Path, result: dict[str, Any]) -> tuple[str, str]:
    return str(result.get("completed_at", "")), str(result_path)


def _status_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        summary[status] = summary.get(status, 0) + 1
    return summary


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


def _markdown_value(value: object) -> str:
    text = str(_csv_value(value))
    return text.replace("|", "\\|")
