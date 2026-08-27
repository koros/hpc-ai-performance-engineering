from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import (
    PlatformCondition,
    discover_experiment_configs,
    load_platform_conditions,
    load_platform_profile,
)
from hpc_ai_perf.results import load_result_records
from hpc_ai_perf.validation import validate_result_records

PLATFORM_TRACKER_COLUMNS = [
    "condition_id",
    "matrix_id",
    "platform",
    "wave",
    "wave_name",
    "status",
    "required_trials",
    "valid_trials",
    "failed_trials",
    "invalid_results",
    "comparability",
    "phase",
    "workload",
    "mode",
    "strategy",
    "gpus",
    "precision",
    "priority",
    "latest_completed_at",
    "result_files",
    "notes",
]


def track_platform_progress(
    platform_profile: str | Path,
    execution_matrix: str | Path = "platforms/execution_matrix.json",
    result_dir: str | Path = "results/raw",
    output_csv: str | Path | None = None,
    output_markdown: str | Path | None = None,
    *,
    config_root: str | Path = "configs",
    wave: str | None = None,
) -> tuple[Path, Path]:
    """Write a condition-level tracker for one execution platform."""
    profile = load_platform_profile(platform_profile)
    conditions = load_platform_conditions(execution_matrix, profile.platform_id, wave=wave)
    rows = build_platform_progress_rows(
        conditions,
        result_dir,
        config_root=config_root,
    )
    suffix = f"_wave_{wave}" if wave else ""
    csv_path = Path(output_csv or f"results/processed/{profile.platform_id}{suffix}_experiment_tracker.csv")
    markdown_path = Path(output_markdown or f"results/processed/{profile.platform_id}{suffix}_experiment_tracker.md")
    write_platform_tracker_csv(rows, csv_path)
    write_platform_tracker_markdown(rows, markdown_path, platform_id=profile.platform_id)
    return csv_path, markdown_path


def build_platform_progress_rows(
    conditions: list[PlatformCondition],
    result_dir: str | Path,
    *,
    config_root: str | Path = "configs",
) -> list[dict[str, Any]]:
    """Combine platform conditions with validated, de-duplicated result trials."""
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs(config_root)}
    invalid_paths = {issue.path.resolve() for issue in validate_result_records(result_dir)}
    indexed_results: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for result_path, result in load_result_records(result_dir):
        condition_id = _result_condition_id(result)
        if condition_id:
            indexed_results.setdefault(condition_id, []).append((result_path, result))

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        entry = entries[condition.matrix_id]
        records = indexed_results.get(condition.condition_id, [])
        valid_records = [(path, data) for path, data in records if path.resolve() not in invalid_paths]
        latest_trials = _latest_records_by_trial(valid_records)
        valid_trials = sum(str(data.get("status", "")).lower() == "completed" for _, data in latest_trials)
        failed_trials = sum(str(data.get("status", "")).lower() == "failed" for _, data in latest_trials)
        invalid_results = len(records) - len(valid_records)
        status = _platform_status(condition, valid_trials, failed_trials, invalid_results)
        result_files = "; ".join(str(path) for path, _ in sorted(records, key=lambda item: str(item[0])))
        completed_times = [str(data.get("completed_at", "")) for _, data in valid_records]
        notes = _platform_notes(condition, status, invalid_results)
        config = entry.config
        rows.append(
            {
                "condition_id": condition.condition_id,
                "matrix_id": condition.matrix_id,
                "platform": condition.platform_id,
                "wave": condition.wave,
                "wave_name": condition.wave_name,
                "status": status,
                "required_trials": condition.required_trials,
                "valid_trials": valid_trials,
                "failed_trials": failed_trials,
                "invalid_results": invalid_results,
                "comparability": condition.comparability,
                "phase": entry.phase,
                "workload": config.workload.name,
                "mode": config.mode,
                "strategy": config.distributed.strategy,
                "gpus": config.hardware.gpus,
                "precision": config.runtime.precision,
                "priority": entry.priority,
                "latest_completed_at": max(completed_times, default=""),
                "result_files": result_files,
                "notes": notes,
            }
        )
    return rows


def write_platform_tracker_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    """Write the platform tracker with a stable CSV schema."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=PLATFORM_TRACKER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PLATFORM_TRACKER_COLUMNS})
    return csv_path


def write_platform_tracker_markdown(
    rows: list[dict[str, Any]], output_markdown: str | Path, *, platform_id: str
) -> Path:
    """Write a concise platform status report for cluster operation."""
    markdown_path = Path(output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        summary[status] = summary.get(status, 0) + 1
    lines = [
        f"# {platform_id.upper()} Experiment Tracker",
        "",
        "Generated from the platform execution matrix and validated raw result evidence.",
        "",
        "## Summary",
        "",
        "| Status | Conditions |",
        "| --- | ---: |",
        *[f"| {status} | {count} |" for status, count in sorted(summary.items())],
        "",
        "## Conditions",
        "",
        "| Condition | Wave | Status | Trials | Workload | GPUs | Precision | Comparability | Notes |",
        "| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        trials = f"{row['valid_trials']}/{row['required_trials']}"
        lines.append(
            "| {condition_id} | {wave} | {status} | {trials} | {workload} | {gpus} | {precision} | "
            "{comparability} | {notes} |".format(
                trials=trials,
                **{key: _markdown_value(value) for key, value in row.items()},
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def _result_condition_id(result: dict[str, Any]) -> str:
    configured = str(result.get("condition_id", "")).strip()
    if configured:
        return configured
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    platform_id = str(result.get("platform_id") or metadata.get("execution_platform") or "").strip()
    experiment_id = str(result.get("experiment_id", "")).strip()
    return f"{experiment_id}@{platform_id}" if experiment_id and platform_id else ""


def _latest_records_by_trial(
    records: list[tuple[Path, dict[str, Any]]],
) -> list[tuple[Path, dict[str, Any]]]:
    latest: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, data in records:
        repetition = data.get("repetition_number")
        trial_key = f"repetition:{repetition}" if repetition is not None else f"run:{data.get('run_id') or path}"
        current = latest.get(trial_key)
        if current is None or _result_sort_key(path, data) > _result_sort_key(*current):
            latest[trial_key] = (path, data)
    return list(latest.values())


def _result_sort_key(path: Path, result: dict[str, Any]) -> tuple[str, str]:
    return str(result.get("completed_at", "")), str(path)


def _platform_status(condition: PlatformCondition, valid_trials: int, failed_trials: int, invalid_results: int) -> str:
    if condition.status == "deferred":
        return "Deferred"
    if valid_trials >= condition.required_trials:
        return "Complete"
    if valid_trials:
        return "Partial"
    if failed_trials:
        return "Failed"
    if invalid_results:
        return "Invalid"
    return "Ready"


def _platform_notes(condition: PlatformCondition, status: str, invalid_results: int) -> str:
    notes: list[str] = []
    if condition.reason:
        notes.append(condition.reason)
    if invalid_results:
        notes.append(f"{invalid_results} result file(s) failed validation")
    if status == "Failed":
        notes.append("latest recorded trial failed; inspect the raw result error")
    return "; ".join(notes)


def _markdown_value(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|")
