from __future__ import annotations

import csv
import statistics
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
        "actual_gpu_hours",
        "energy_joules",
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
        "actual_gpu_hours",
        "throughput_tokens_per_gpu",
        "throughput_samples_per_gpu",
        "tokens_per_estimated_gpu_hour",
        "samples_per_estimated_gpu_hour",
        "tokens_per_actual_gpu_hour",
        "samples_per_actual_gpu_hour",
        "energy_joules",
        "tokens_per_kwh",
    ],
    "scaling_summary": [
        "phase",
        "workload",
        "model",
        "precision",
        "strategy",
        "scaling_type",
        "gpus",
        "runtime_seconds",
        "baseline_gpus",
        "baseline_runtime_seconds",
        "speedup",
        "scaling_efficiency",
        "throughput_tokens_sec",
        "throughput_samples_sec",
        "trials",
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
        "avg_quality_loss",
        "avg_quality_perplexity",
        "avg_quality_accuracy",
    ],
    "trial_summary": [
        "experiment_id",
        "phase",
        "workload",
        "mode",
        "completed_trials",
        "failed_trials",
        "runtime_mean_seconds",
        "runtime_std_seconds",
        "runtime_cv_percent",
        "throughput_tokens_mean",
        "throughput_tokens_std",
        "throughput_samples_mean",
        "throughput_samples_std",
    ],
    "strategy_summary": [
        "workload",
        "model",
        "precision",
        "strategy",
        "gpus",
        "trials",
        "avg_throughput_tokens_sec",
        "avg_throughput_samples_sec",
        "max_memory_used_gb",
        "avg_step_time_ms",
        "avg_loss",
    ],
    "communication_summary": [
        "experiment_id",
        "workload",
        "strategy",
        "gpus",
        "trials",
        "avg_allreduce_time_seconds",
        "avg_nccl_bandwidth_gbps",
        "avg_throughput_tokens_sec",
        "communication_to_compute_ratio",
    ],
    "data_movement_summary": [
        "experiment_id",
        "workload",
        "num_workers",
        "pinned_memory",
        "prefetch_factor",
        "trials",
        "avg_data_loading_seconds",
        "avg_transfer_seconds",
        "avg_measured_seconds",
        "avg_throughput_tokens_sec",
    ],
    "memory_summary": [
        "experiment_id",
        "workload",
        "study_type",
        "activation_checkpointing",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "effective_global_batch_size",
        "trials",
        "avg_throughput_tokens_sec",
        "max_memory_used_gb",
        "max_memory_reserved_gb",
        "avg_step_time_ms",
        "avg_loss",
    ],
    "inference_summary": [
        "experiment_id",
        "workload",
        "model",
        "platform",
        "framework",
        "precision",
        "gpus",
        "batch_size",
        "concurrent_requests",
        "prompt_length",
        "trials",
        "avg_throughput_tokens_sec",
        "avg_throughput_samples_sec",
        "avg_latency_p50_ms",
        "avg_latency_p95_ms",
        "avg_ttft_p50_ms",
        "avg_queueing_p50_ms",
        "avg_memory_used_gb",
        "avg_energy_joules",
    ],
    "hardware_summary": [
        "workload",
        "model",
        "mode",
        "platform",
        "gpu_model",
        "precision",
        "trials",
        "avg_throughput_tokens_sec",
        "avg_throughput_samples_sec",
        "avg_latency_p50_ms",
        "avg_memory_used_gb",
        "avg_energy_joules",
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
        "trial_summary": build_trial_summary(rows),
        "strategy_summary": build_strategy_summary(rows),
        "communication_summary": build_communication_summary(rows),
        "data_movement_summary": build_data_movement_summary(rows),
        "memory_summary": build_memory_summary(rows),
        "inference_summary": build_inference_summary(rows),
        "hardware_summary": build_hardware_summary(rows),
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
        completed_rows = [row for row in group_rows if row.get("status") == "completed"]
        runtime_values = [_float(row.get("runtime_seconds")) for row in completed_rows]
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
                "avg_throughput_tokens_sec": _mean(_metric_values(completed_rows, "throughput_tokens_sec")),
                "avg_throughput_samples_sec": _mean(_metric_values(completed_rows, "throughput_samples_sec")),
                "avg_gpu_utilization": _mean(_metric_values(completed_rows, "gpu_utilization")),
                "max_memory_used_gb": _max(_metric_values(completed_rows, "memory_used_gb")),
                "estimated_gpu_hours": _sum(_metric_values(group_rows, "gpu_hours_estimate")),
                "actual_gpu_hours": _sum(_metric_values(completed_rows, "actual_gpu_hours")),
                "energy_joules": _sum(_metric_values(completed_rows, "energy_joules")),
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
        if row.get("status") != "completed":
            continue
        gpus = _float(row.get("gpus"))
        gpu_hours = _float(row.get("gpu_hours_estimate"))
        actual_gpu_hours = _first_float(row, "actual_gpu_hours", "metric_gpu_hours")
        total_tokens = _first_float(row, "metric_total_tokens")
        total_samples = _first_float(row, "metric_total_samples", "metric_samples")
        energy_joules = _first_float(row, "energy_joules", "metric_energy_joules_estimate")
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
                "actual_gpu_hours": actual_gpu_hours,
                "throughput_tokens_per_gpu": _divide(throughput_tokens, gpus),
                "throughput_samples_per_gpu": _divide(throughput_samples, gpus),
                "tokens_per_estimated_gpu_hour": _divide(total_tokens, gpu_hours),
                "samples_per_estimated_gpu_hour": _divide(total_samples, gpu_hours),
                "tokens_per_actual_gpu_hour": _divide(total_tokens, actual_gpu_hours),
                "samples_per_actual_gpu_hour": _divide(total_samples, actual_gpu_hours),
                "energy_joules": energy_joules,
                "tokens_per_kwh": _divide(total_tokens, _divide(energy_joules, 3_600_000)),
            }
        )

    return sorted(rankings, key=lambda item: _sort_float(item.get("throughput_tokens_sec")), reverse=True)


def build_scaling_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build speedup and scaling-efficiency rows for completed experiments.

    For each phase/workload/model/precision/strategy group, the smallest GPU
    count with a runtime becomes the baseline. Other rows are compared against
    that baseline to estimate speedup and efficiency.
    """
    completed_rows = [
        row
        for row in rows
        if row.get("status") == "completed"
        and (row.get("scaling_type") in {"strong", "weak"} or row.get("phase") in {"Strong scaling", "Weak scaling"})
    ]
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in completed_rows:
        key = (
            _value(row, "phase"),
            _value(row, "workload"),
            _value(row, "model"),
            _value(row, "precision"),
            _value(row, "strategy"),
            _value(row, "scaling_type") or ("weak" if _value(row, "phase") == "Weak scaling" else "strong"),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (phase, workload, model, precision, strategy, scaling_type), group_rows in sorted(grouped.items()):
        by_gpus: dict[float, list[dict[str, str]]] = defaultdict(list)
        for row in group_rows:
            gpus = _float(row.get("gpus"))
            measured = _first_float(row, "metric_measured_seconds", "runtime_seconds")
            if gpus is not None and measured is not None:
                by_gpus[gpus].append(row)
        if not by_gpus:
            continue
        baseline_gpus = min(by_gpus)
        baseline_runtime = _mean(
            [_first_float(row, "metric_measured_seconds", "runtime_seconds") for row in by_gpus[baseline_gpus]]
        )
        baseline_tokens = _mean(
            [
                _first_float(row, "throughput_tokens_sec", "metric_throughput_tokens_sec")
                for row in by_gpus[baseline_gpus]
            ]
        )
        baseline_samples = _mean(
            [
                _first_float(row, "throughput_samples_sec", "metric_throughput_samples_sec")
                for row in by_gpus[baseline_gpus]
            ]
        )

        for gpus, gpu_rows in sorted(by_gpus.items()):
            runtime = _mean([_first_float(row, "metric_measured_seconds", "runtime_seconds") for row in gpu_rows])
            throughput_tokens = _mean(
                [_first_float(row, "throughput_tokens_sec", "metric_throughput_tokens_sec") for row in gpu_rows]
            )
            throughput_samples = _mean(
                [_first_float(row, "throughput_samples_sec", "metric_throughput_samples_sec") for row in gpu_rows]
            )
            gpu_ratio = _divide(gpus, baseline_gpus)
            if scaling_type == "weak":
                speedup = _divide(throughput_tokens, baseline_tokens) or _divide(throughput_samples, baseline_samples)
                efficiency = _divide(baseline_runtime, runtime)
            else:
                speedup = _divide(baseline_runtime, runtime)
                efficiency = _divide(speedup, gpu_ratio)
            summaries.append(
                {
                    "phase": phase,
                    "workload": workload,
                    "model": model,
                    "precision": precision,
                    "strategy": strategy,
                    "scaling_type": scaling_type,
                    "gpus": gpus,
                    "runtime_seconds": runtime,
                    "baseline_gpus": baseline_gpus,
                    "baseline_runtime_seconds": baseline_runtime,
                    "speedup": speedup,
                    "scaling_efficiency": efficiency,
                    "throughput_tokens_sec": throughput_tokens,
                    "throughput_samples_sec": throughput_samples,
                    "trials": len(gpu_rows),
                }
            )

    return summaries


def build_precision_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize average runtime, throughput, memory, and latency by precision."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed":
            continue
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
                "avg_quality_loss": _mean(_metric_values(group_rows, "quality_loss")),
                "avg_quality_perplexity": _mean(_metric_values(group_rows, "quality_perplexity")),
                "avg_quality_accuracy": _mean(_metric_values(group_rows, "quality_accuracy")),
            }
        )

    return summaries


def build_trial_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Quantify independent-run variance without mixing failed trials."""
    grouped = _group_rows(rows, "experiment_id", "phase", "workload", "mode")
    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        completed = [row for row in group_rows if row.get("status") == "completed"]
        runtime = _numeric(_metric_values(completed, "runtime_seconds"))
        token_throughput = _numeric(_metric_values(completed, "throughput_tokens_sec"))
        sample_throughput = _numeric(_metric_values(completed, "throughput_samples_sec"))
        summaries.append(
            {
                "experiment_id": key[0],
                "phase": key[1],
                "workload": key[2],
                "mode": key[3],
                "completed_trials": len(completed),
                "failed_trials": sum(row.get("status") == "failed" for row in group_rows),
                "runtime_mean_seconds": _mean(runtime),
                "runtime_std_seconds": _std(runtime),
                "runtime_cv_percent": _cv_percent(runtime),
                "throughput_tokens_mean": _mean(token_throughput),
                "throughput_tokens_std": _std(token_throughput),
                "throughput_samples_mean": _mean(sample_throughput),
                "throughput_samples_std": _std(sample_throughput),
            }
        )
    return summaries


def build_strategy_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare throughput-to-memory trade-offs for distributed strategies."""
    selected = _completed_phase(rows, "Distributed training strategy")
    grouped = _group_rows(selected, "workload", "model", "precision", "strategy", "gpus")
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "workload": key[0],
                "model": key[1],
                "precision": key[2],
                "strategy": key[3],
                "gpus": key[4],
                "trials": len(group_rows),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                "avg_throughput_samples_sec": _mean(_metric_values(group_rows, "throughput_samples_sec")),
                "max_memory_used_gb": _max(_metric_values(group_rows, "memory_used_gb")),
                "avg_step_time_ms": _mean(_metric_values(group_rows, "step_time_mean_ms")),
                "avg_loss": _mean(_metric_values(group_rows, "loss")),
            }
        )
    return summaries


def build_communication_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Expose collective latency and bandwidth for RQ4."""
    grouped = _group_rows(
        _completed_phase(rows, "Communication analysis"),
        "experiment_id",
        "workload",
        "strategy",
        "gpus",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        communication = _mean(_metric_values(group_rows, "allreduce_time_seconds"))
        integrated_communication = _mean(_metric_values(group_rows, "communication_time_seconds"))
        measured = _mean(_metric_values(group_rows, "measured_seconds"))
        summaries.append(
            {
                "experiment_id": key[0],
                "workload": key[1],
                "strategy": key[2],
                "gpus": key[3],
                "trials": len(group_rows),
                "avg_allreduce_time_seconds": communication,
                "avg_nccl_bandwidth_gbps": _mean(_metric_values(group_rows, "nccl_bandwidth_gbps")),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                # Standalone AllReduce time is not an integrated training-time
                # attribution. Leave the ratio empty until an in-loop
                # communication timer is available.
                "communication_to_compute_ratio": _divide(integrated_communication, measured),
            }
        )
    return summaries


def build_data_movement_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize configured DataLoader factors and measured pipeline time."""
    grouped = _group_rows(_completed_phase(rows, "Data movement"), "experiment_id", "workload")
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summaries.append(
            {
                "experiment_id": key[0],
                "workload": key[1],
                "num_workers": first.get("parameter_num_workers", ""),
                "pinned_memory": first.get("parameter_pinned_memory", ""),
                "prefetch_factor": first.get("parameter_prefetch_factor", ""),
                "trials": len(group_rows),
                "avg_data_loading_seconds": _mean(_metric_values(group_rows, "data_loading_seconds")),
                "avg_transfer_seconds": _mean(_metric_values(group_rows, "cpu_to_gpu_transfer_seconds")),
                "avg_measured_seconds": _mean(_metric_values(group_rows, "measured_seconds")),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
            }
        )
    return summaries


def build_inference_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize serving-load factors and latency distributions."""
    selected = [row for row in rows if row.get("status") == "completed" and row.get("mode") == "inference"]
    grouped = _group_rows(
        selected,
        "experiment_id",
        "workload",
        "model",
        "platform",
        "framework",
        "precision",
        "gpus",
        "batch_size",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summaries.append(
            {
                "experiment_id": key[0],
                "workload": key[1],
                "model": key[2],
                "platform": key[3],
                "framework": key[4],
                "precision": key[5],
                "gpus": key[6],
                "batch_size": key[7],
                "concurrent_requests": first.get("parameter_concurrent_requests", ""),
                "prompt_length": first.get("parameter_prompt_length", ""),
                "trials": len(group_rows),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                "avg_throughput_samples_sec": _mean(_metric_values(group_rows, "throughput_samples_sec")),
                "avg_latency_p50_ms": _mean(_metric_values(group_rows, "latency_p50_ms")),
                "avg_latency_p95_ms": _mean(_metric_values(group_rows, "latency_p95_ms")),
                "avg_ttft_p50_ms": _mean(_metric_values(group_rows, "time_to_first_token_p50_ms")),
                "avg_queueing_p50_ms": _mean(_metric_values(group_rows, "queueing_time_p50_ms")),
                "avg_memory_used_gb": _mean(_metric_values(group_rows, "memory_used_gb")),
                "avg_energy_joules": _mean(_metric_values(group_rows, "energy_joules")),
            }
        )
    return summaries


def build_memory_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Summarize checkpointing, micro-batch, and accumulation trade-offs."""
    grouped = _group_rows(_completed_phase(rows, "Memory optimisation"), "experiment_id", "workload")
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summaries.append(
            {
                "experiment_id": key[0],
                "workload": key[1],
                "study_type": first.get("parameter_memory_study_type", ""),
                "activation_checkpointing": first.get("parameter_activation_checkpointing", ""),
                "micro_batch_size": first.get("parameter_micro_batch_size", ""),
                "gradient_accumulation_steps": first.get("parameter_gradient_accumulation_steps", ""),
                "effective_global_batch_size": first.get("global_batch_size", ""),
                "trials": len(group_rows),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                "max_memory_used_gb": _max(_metric_values(group_rows, "memory_used_gb")),
                "max_memory_reserved_gb": _max(_metric_values(group_rows, "memory_reserved_gb")),
                "avg_step_time_ms": _mean(_metric_values(group_rows, "step_time_mean_ms")),
                "avg_loss": _mean(_metric_values(group_rows, "loss")),
            }
        )
    return summaries


def build_hardware_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare completed H100/L40S experiments using observed platform fields."""
    selected = _completed_phase(rows, "Hardware comparison")
    grouped = _group_rows(selected, "workload", "model", "mode", "platform", "gpu_model", "precision")
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "workload": key[0],
                "model": key[1],
                "mode": key[2],
                "platform": key[3],
                "gpu_model": key[4],
                "precision": key[5],
                "trials": len(group_rows),
                "avg_throughput_tokens_sec": _mean(_metric_values(group_rows, "throughput_tokens_sec")),
                "avg_throughput_samples_sec": _mean(_metric_values(group_rows, "throughput_samples_sec")),
                "avg_latency_p50_ms": _mean(_metric_values(group_rows, "latency_p50_ms")),
                "avg_memory_used_gb": _mean(_metric_values(group_rows, "memory_used_gb")),
                "avg_energy_joules": _mean(_metric_values(group_rows, "energy_joules")),
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


def _completed_phase(rows: list[dict[str, str]], phase: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "completed" and row.get("phase") == phase]


def _group_rows(rows: list[dict[str, str]], *names: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(_value(row, name) for name in names)].append(row)
    return grouped


def _numeric(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _std(values: list[float | None]) -> float | None:
    numeric_values = _numeric(values)
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return 0.0
    return statistics.stdev(numeric_values)


def _cv_percent(values: list[float | None]) -> float | None:
    mean = _mean(values)
    standard_deviation = _std(values)
    if mean in {None, 0} or standard_deviation is None:
        return None
    return standard_deviation / mean * 100


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
