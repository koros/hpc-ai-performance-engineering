from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

ANALYSIS_TABLES = {
    "phase_summary": [
        "phase",
        "priority",
        "mode",
        "workload",
        "experiments",
        "completed",
        "failed",
        "total_runtime_seconds",
        "avg_runtime_seconds",
        "avg_throughput_tokens_sec",
        "avg_throughput_samples_sec",
        "avg_gpu_utilization",
        "max_memory_used_gb",
        "estimated_gpu_hours",
    ],
    "experiment_rankings": [
        "experiment_id",
        "phase",
        "priority",
        "workload",
        "model",
        "mode",
        "gpus",
        "precision",
        "strategy",
        "status",
        "throughput_tokens_sec",
        "throughput_samples_sec",
        "runtime_seconds",
        "gpu_hours_estimate",
        "throughput_tokens_per_gpu",
        "throughput_samples_per_gpu",
        "tokens_per_estimated_gpu_hour",
        "samples_per_estimated_gpu_hour",
    ],
    "scaling_summary": [
        "phase",
        "workload",
        "model",
        "precision",
        "strategy",
        "gpus",
        "runtime_seconds",
        "baseline_gpus",
        "baseline_runtime_seconds",
        "speedup",
        "scaling_efficiency",
        "throughput_tokens_sec",
        "throughput_samples_sec",
    ],
    "precision_summary": [
        "workload",
        "model",
        "mode",
        "precision",
        "experiments",
        "avg_runtime_seconds",
        "avg_throughput_tokens_sec",
        "avg_throughput_samples_sec",
        "avg_memory_used_gb",
        "avg_latency_p50_ms",
        "avg_latency_p95_ms",
    ],
}


def analyze_results(
    input_csv: str | Path = "results/processed/experiments.csv", output_dir: str | Path = "results/analysis"
) -> list[Path]:
    """Build all derived analysis CSV tables from the processed results CSV.

    The dashboard and dissertation analysis use these smaller tables instead of
    recalculating summaries from raw records every time. Each table has a fixed
    schema declared in `ANALYSIS_TABLES`, which keeps refresh output predictable.
    """
    rows = load_processed_results(input_csv)
    tables = build_analysis_tables(rows)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for table_name, table_rows in tables.items():
        table_path = output_root / f"{table_name}.csv"
        _write_table(table_path, ANALYSIS_TABLES[table_name], table_rows)
        written_paths.append(table_path)

    return written_paths


def load_processed_results(input_csv: str | Path) -> list[dict[str, str]]:
    """Load the processed results CSV produced by aggregation."""
    csv_path = Path(input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed results CSV not found: {csv_path}")

    with csv_path.open(encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def build_analysis_tables(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    """Build every named analysis table from processed result rows."""
    return {
        "phase_summary": build_phase_summary(rows),
        "experiment_rankings": build_experiment_rankings(rows),
        "scaling_summary": build_scaling_summary(rows),
        "precision_summary": build_precision_summary(rows),
    }


def build_phase_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize runtime, throughput, utilization, and cost by phase grouping.

    Rows are grouped by phase, priority, mode, and workload so the output can
    answer questions like "how much baseline profiling has completed?" or "which
    workload phase is consuming the most estimated GPU-hours?"
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(_value(row, "phase"), _value(row, "priority"), _value(row, "mode"), _value(row, "workload"))].append(
            row
        )

    summaries: list[dict[str, Any]] = []
    for (phase, priority, mode, workload), group_rows in sorted(grouped.items()):
        runtime_values = [_float(row.get("runtime_seconds")) for row in group_rows]
        summaries.append(
            {
                "phase": phase,
                "priority": priority,
                "mode": mode,
                "workload": workload,
                "experiments": len(group_rows),
                "completed": sum(1 for row in group_rows if row.get("status") == "completed"),
                "failed": sum(1 for row in group_rows if row.get("status") == "failed"),
                "total_runtime_seconds": _sum(runtime_values),
                "avg_runtime_seconds": _mean(runtime_values),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                "avg_throughput_samples_sec": _mean(_metric_values(group_rows, "throughput_samples_sec")),
                "avg_gpu_utilization": _mean(_metric_values(group_rows, "gpu_utilization")),
                "max_memory_used_gb": _max(_metric_values(group_rows, "memory_used_gb")),
                "estimated_gpu_hours": _sum(_metric_values(group_rows, "gpu_hours_estimate")),
            }
        )

    return summaries


def build_experiment_rankings(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build per-experiment ranking rows with normalized efficiency metrics.

    The ranking table keeps raw throughput and runtime values, then adds
    throughput-per-GPU and throughput-per-estimated-GPU-hour where the necessary
    denominators are available.
    """
    rankings: list[dict[str, Any]] = []
    for row in rows:
        gpus = _float(row.get("gpus"))
        gpu_hours = _float(row.get("gpu_hours_estimate"))
        throughput_tokens = _first_float(row, "throughput_tokens_sec", "metric_throughput_tokens_sec")
        throughput_samples = _first_float(row, "throughput_samples_sec", "metric_throughput_samples_sec")

        rankings.append(
            {
                "experiment_id": row.get("experiment_id", ""),
                "phase": row.get("phase", ""),
                "priority": row.get("priority", ""),
                "workload": row.get("workload", ""),
                "model": row.get("model", ""),
                "mode": row.get("mode", ""),
                "gpus": row.get("gpus", ""),
                "precision": row.get("precision", ""),
                "strategy": row.get("strategy", ""),
                "status": row.get("status", ""),
                "throughput_tokens_sec": throughput_tokens,
                "throughput_samples_sec": throughput_samples,
                "runtime_seconds": _float(row.get("runtime_seconds")),
                "gpu_hours_estimate": gpu_hours,
                "throughput_tokens_per_gpu": _divide(throughput_tokens, gpus),
                "throughput_samples_per_gpu": _divide(throughput_samples, gpus),
                "tokens_per_estimated_gpu_hour": _divide(_scale(throughput_tokens, 3600), gpu_hours),
                "samples_per_estimated_gpu_hour": _divide(
                    _scale(throughput_samples, 3600),
                    gpu_hours,
                ),
            }
        )

    return sorted(rankings, key=lambda item: _sort_float(item.get("throughput_tokens_sec")), reverse=True)


def build_scaling_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build speedup and scaling-efficiency rows for completed experiments.

    For each phase/workload/model/precision/strategy group, the smallest GPU
    count with a runtime becomes the baseline. Other rows are compared against
    that baseline to estimate speedup and efficiency.
    """
    completed_rows = [row for row in rows if row.get("status") == "completed"]
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed_rows:
        key = (
            _value(row, "phase"),
            _value(row, "workload"),
            _value(row, "model"),
            _value(row, "precision"),
            _value(row, "strategy"),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (phase, workload, model, precision, strategy), group_rows in sorted(grouped.items()):
        runtime_rows = [row for row in group_rows if _float(row.get("runtime_seconds")) is not None]
        if not runtime_rows:
            continue
        baseline = min(runtime_rows, key=lambda row: _float(row.get("gpus")) or 0)
        baseline_gpus = _float(baseline.get("gpus"))
        baseline_runtime = _float(baseline.get("runtime_seconds"))

        for row in sorted(runtime_rows, key=lambda item: _float(item.get("gpus")) or 0):
            gpus = _float(row.get("gpus"))
            runtime = _float(row.get("runtime_seconds"))
            speedup = _divide(baseline_runtime, runtime)
            gpu_ratio = _divide(gpus, baseline_gpus)
            summaries.append(
                {
                    "phase": phase,
                    "workload": workload,
                    "model": model,
                    "precision": precision,
                    "strategy": strategy,
                    "gpus": row.get("gpus", ""),
                    "runtime_seconds": runtime,
                    "baseline_gpus": baseline.get("gpus", ""),
                    "baseline_runtime_seconds": baseline_runtime,
                    "speedup": speedup,
                    "scaling_efficiency": _divide(speedup, gpu_ratio),
                    "throughput_tokens_sec": _first_float(row, "throughput_tokens_sec", "metric_throughput_tokens_sec"),
                    "throughput_samples_sec": _first_float(
                        row, "throughput_samples_sec", "metric_throughput_samples_sec"
                    ),
                }
            )

    return summaries


def build_precision_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize average runtime, throughput, memory, and latency by precision."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(_value(row, "workload"), _value(row, "model"), _value(row, "mode"), _value(row, "precision"))].append(
            row
        )

    summaries: list[dict[str, Any]] = []
    for (workload, model, mode, precision), group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "workload": workload,
                "model": model,
                "mode": mode,
                "precision": precision,
                "experiments": len(group_rows),
                "avg_runtime_seconds": _mean(_metric_values(group_rows, "runtime_seconds")),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                "avg_throughput_samples_sec": _mean(_metric_values(group_rows, "throughput_samples_sec")),
                "avg_memory_used_gb": _mean(_metric_values(group_rows, "memory_used_gb")),
                "avg_latency_p50_ms": _mean(_metric_values(group_rows, "latency_p50_ms")),
                "avg_latency_p95_ms": _mean(_metric_values(group_rows, "latency_p95_ms")),
            }
        )

    return summaries


def _write_table(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write one analysis table with a stable field order."""
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _metric_values(rows: list[dict[str, str]], name: str) -> list[float | None]:
    """Collect numeric values from stable columns or matching metric-prefixed columns."""
    return [_first_float(row, name, f"metric_{name}") for row in rows]


def _first_float(row: dict[str, str], *names: str) -> float | None:
    """Return the first parseable float from a set of candidate column names."""
    for name in names:
        value = _float(row.get(name))
        if value is not None:
            return value
    return None


def _float(value: object) -> float | None:
    """Parse a CSV cell into a float, treating blanks and invalid values as missing."""
    if value in {None, ""}:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _mean(values: list[float | None]) -> float | None:
    """Return the arithmetic mean of present numeric values."""
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _sum(values: list[float | None]) -> float:
    """Return the sum of present numeric values, ignoring missing values."""
    return sum(value for value in values if value is not None)


def _max(values: list[float | None]) -> float | None:
    """Return the maximum present numeric value, or null when no value exists."""
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    """Safely divide two optional numbers without producing divide-by-zero errors."""
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _scale(value: float | None, factor: float) -> float | None:
    """Multiply an optional numeric value by a constant factor."""
    if value is None:
        return None
    return value * factor


def _value(row: dict[str, str], name: str) -> str:
    """Return a string cell value, normalising missing values to an empty string."""
    return row.get(name, "") or ""


def _sort_float(value: object) -> float:
    """Return a sortable float, placing missing values at the end of descending lists."""
    numeric_value = _float(value)
    if numeric_value is None:
        return float("-inf")
    return numeric_value


def _csv_value(value: object) -> object:
    """Convert missing analysis values into empty CSV cells."""
    if value is None:
        return ""
    return value
