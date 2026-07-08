from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

BASE_COLUMNS = [
    "schema_version",
    "experiment_id",
    "matrix_id",
    "experiment",
    "phase",
    "priority",
    "workload",
    "model",
    "mode",
    "status",
    "gpus",
    "gpu_model",
    "platform",
    "strategy",
    "precision",
    "started_at",
    "completed_at",
    "runtime_seconds",
    "throughput_tokens_sec",
    "throughput_samples_sec",
    "gpu_utilization",
    "memory_used_gb",
    "latency_p50_ms",
    "latency_p95_ms",
    "gpu_hours_estimate",
    "source_path",
    "error",
]


def aggregate_results(
    input_dir: str | Path = "results/raw", output_path: str | Path = "results/processed/experiments.csv"
) -> Path:
    rows = [
        flatten_result_record(record, source_path=source_path) for source_path, record in load_result_records(input_dir)
    ]
    return write_results_csv(rows, output_path)


def load_result_records(input_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for result_path in sorted(Path(input_dir).rglob("*.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Result file must contain a JSON object: {result_path}")
        if _looks_like_result_record(data):
            records.append((result_path, data))
    return records


def flatten_result_record(record: dict[str, Any], source_path: str | Path | None = None) -> dict[str, Any]:
    config = _mapping(record.get("config"))
    metadata = _mapping(config.get("metadata"))
    hardware = _mapping(config.get("hardware"))
    distributed = _mapping(config.get("distributed"))

    row: dict[str, Any] = {
        "schema_version": record.get("schema_version"),
        "experiment_id": record.get("experiment_id"),
        "matrix_id": metadata.get("matrix_id", record.get("experiment_id")),
        "experiment": record.get("experiment"),
        "phase": metadata.get("phase"),
        "priority": metadata.get("priority"),
        "workload": record.get("workload"),
        "model": record.get("model"),
        "mode": record.get("mode"),
        "status": record.get("status"),
        "gpus": record.get("gpus"),
        "gpu_model": hardware.get("gpu_model"),
        "platform": hardware.get("platform"),
        "strategy": distributed.get("strategy"),
        "precision": record.get("precision"),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
        "runtime_seconds": record.get("runtime_seconds"),
        "throughput_tokens_sec": record.get("throughput_tokens_sec"),
        "throughput_samples_sec": record.get("throughput_samples_sec"),
        "gpu_utilization": record.get("gpu_utilization"),
        "memory_used_gb": record.get("memory_used_gb"),
        "latency_p50_ms": record.get("latency_p50_ms"),
        "latency_p95_ms": record.get("latency_p95_ms"),
        "gpu_hours_estimate": metadata.get("gpu_hours_estimate"),
        "source_path": str(source_path) if source_path else None,
        "error": record.get("error"),
    }

    for key, value in _mapping(record.get("metrics")).items():
        row[f"metric_{_slugify(key)}"] = value
    for key, value in _mapping(record.get("environment")).items():
        row[f"environment_{_slugify(key)}"] = value

    return row


def write_results_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted({key for row in rows for key in row if key not in BASE_COLUMNS})
    fieldnames = [*BASE_COLUMNS, *extra_columns]

    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})

    return csv_path


def _looks_like_result_record(data: dict[str, Any]) -> bool:
    return "experiment_id" in data and "status" in data and "runtime_seconds" in data


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "value"


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    return value
