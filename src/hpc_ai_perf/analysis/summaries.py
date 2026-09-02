from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from hpc_ai_perf.result_identity import COMPARABILITY_VALUES, MATRIX_PLATFORMS, is_matrix_experiment_id

ANALYSIS_TABLES = {
    "phase_summary": [
        "platform_id",
        "comparability",
        "phase",
        "priority",
        "mode",
        "workload",
        "throughput_unit",
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "phase",
        "priority",
        "workload",
        "model",
        "mode",
        "gpus",
        "precision",
        "strategy",
        "status",
        "throughput_unit",
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
        "energy_scope",
        "tokens_per_kwh",
    ],
    "scaling_summary": [
        "platform_id",
        "comparability",
        "phase",
        "workload",
        "model",
        "precision",
        "strategy",
        "scaling_type",
        "throughput_unit",
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
        "platform_id",
        "comparability",
        "workload",
        "model",
        "mode",
        "precision",
        "throughput_unit",
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "phase",
        "workload",
        "mode",
        "throughput_unit",
        "required_trials",
        "completed_trials",
        "failed_trials",
        "evidence_status",
        "outliers_excluded",
        "runtime_mean_seconds",
        "runtime_median_seconds",
        "runtime_std_seconds",
        "runtime_cv_percent",
        "runtime_ci95_low_seconds",
        "runtime_ci95_high_seconds",
        "throughput_tokens_mean",
        "throughput_tokens_median",
        "throughput_tokens_std",
        "throughput_tokens_ci95_low",
        "throughput_tokens_ci95_high",
        "throughput_samples_mean",
        "throughput_samples_median",
        "throughput_samples_std",
        "throughput_samples_ci95_low",
        "throughput_samples_ci95_high",
    ],
    "strategy_summary": [
        "platform_id",
        "comparability",
        "workload",
        "model",
        "precision",
        "strategy",
        "throughput_unit",
        "gpus",
        "trials",
        "avg_throughput_tokens_sec",
        "avg_throughput_samples_sec",
        "max_memory_used_gb",
        "avg_step_time_ms",
        "avg_loss",
    ],
    "communication_summary": [
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "throughput_unit",
        "strategy",
        "gpus",
        "trials",
        "avg_allreduce_time_seconds",
        "avg_nccl_bandwidth_gbps",
        "avg_throughput_tokens_sec",
        "communication_to_compute_ratio",
    ],
    "data_movement_summary": [
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "throughput_unit",
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "throughput_unit",
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "model",
        "platform",
        "framework",
        "precision",
        "gpus",
        "batch_size",
        "concurrent_requests",
        "prompt_length",
        "throughput_unit",
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "model",
        "mode",
        "platform",
        "gpu_model",
        "precision",
        "throughput_unit",
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
    _validate_analysis_identities(rows)
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
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                _value(row, "platform_id"),
                _value(row, "comparability"),
                _value(row, "phase"),
                _value(row, "priority"),
                _value(row, "mode"),
                _value(row, "workload"),
                _value(row, "throughput_unit"),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (platform_id, comparability, phase, priority, mode, workload, throughput_unit), group_rows in sorted(
        grouped.items()
    ):
        completed_rows = [row for row in group_rows if row.get("status") == "completed"]
        runtime_values = [_float(row.get("runtime_seconds")) for row in completed_rows]
        summaries.append(
            {
                "platform_id": platform_id,
                "comparability": comparability,
                "phase": phase,
                "priority": priority,
                "mode": mode,
                "workload": workload,
                "throughput_unit": throughput_unit,
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
        energy_scope = _value(row, "energy_scope")
        energy_joules = _first_float(row, "energy_joules") if energy_scope == "measured_region" else None
        throughput_tokens = _first_float(row, "throughput_tokens_sec", "metric_throughput_tokens_sec")
        throughput_samples = _first_float(row, "throughput_samples_sec", "metric_throughput_samples_sec")

        rankings.append(
            {
                "condition_id": row.get("condition_id", row.get("experiment_id", "")),
                "experiment_id": row.get("experiment_id", ""),
                "platform_id": row.get("platform_id", ""),
                "comparability": row.get("comparability", ""),
                "phase": row.get("phase", ""),
                "priority": row.get("priority", ""),
                "workload": row.get("workload", ""),
                "model": row.get("model", ""),
                "mode": row.get("mode", ""),
                "gpus": row.get("gpus", ""),
                "precision": row.get("precision", ""),
                "strategy": row.get("strategy", ""),
                "status": row.get("status", ""),
                "throughput_unit": row.get("throughput_unit", ""),
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
                "energy_scope": energy_scope,
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
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in completed_rows:
        key = (
            _value(row, "platform_id"),
            _value(row, "comparability"),
            _value(row, "phase"),
            _value(row, "workload"),
            _value(row, "model"),
            _value(row, "precision"),
            _value(row, "strategy"),
            _value(row, "scaling_type") or ("weak" if _value(row, "phase") == "Weak scaling" else "strong"),
            _value(row, "throughput_unit"),
        )
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (
        platform_id,
        comparability,
        phase,
        workload,
        model,
        precision,
        strategy,
        scaling_type,
        throughput_unit,
    ), group_rows in sorted(grouped.items()):
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
                    "platform_id": platform_id,
                    "comparability": comparability,
                    "phase": phase,
                    "workload": workload,
                    "model": model,
                    "precision": precision,
                    "strategy": strategy,
                    "scaling_type": scaling_type,
                    "throughput_unit": throughput_unit,
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
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed":
            continue
        grouped[
            (
                _value(row, "platform_id"),
                _value(row, "comparability"),
                _value(row, "workload"),
                _value(row, "model"),
                _value(row, "mode"),
                _value(row, "precision"),
                _value(row, "throughput_unit"),
            )
        ].append(row)

    summaries: list[dict[str, Any]] = []
    for (platform_id, comparability, workload, model, mode, precision, throughput_unit), group_rows in sorted(
        grouped.items()
    ):
        summaries.append(
            {
                "platform_id": platform_id,
                "comparability": comparability,
                "workload": workload,
                "model": model,
                "mode": mode,
                "precision": precision,
                "throughput_unit": throughput_unit,
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
    grouped = _group_rows(
        rows,
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "phase",
        "workload",
        "mode",
        "throughput_unit",
    )
    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        configured_trials = [_float(row.get("required_trials")) for row in group_rows]
        required_trials = int(max((value for value in configured_trials if value is not None), default=3))
        completed = [row for row in group_rows if row.get("status") == "completed"]
        runtime = _numeric(_metric_values(completed, "runtime_seconds"))
        token_throughput = _numeric(_metric_values(completed, "throughput_tokens_sec"))
        sample_throughput = _numeric(_metric_values(completed, "throughput_samples_sec"))
        runtime_ci = _ci95(runtime)
        token_ci = _ci95(token_throughput)
        sample_ci = _ci95(sample_throughput)
        summaries.append(
            {
                "condition_id": key[0],
                "experiment_id": key[1],
                "platform_id": key[2],
                "comparability": key[3],
                "phase": key[4],
                "workload": key[5],
                "mode": key[6],
                "throughput_unit": key[7],
                "required_trials": required_trials,
                "completed_trials": len(completed),
                "failed_trials": sum(row.get("status") == "failed" for row in group_rows),
                "evidence_status": "sufficient" if len(completed) >= required_trials else "insufficient_data",
                "outliers_excluded": 0,
                "runtime_mean_seconds": _mean(runtime),
                "runtime_median_seconds": _median(runtime),
                "runtime_std_seconds": _std(runtime),
                "runtime_cv_percent": _cv_percent(runtime),
                "runtime_ci95_low_seconds": runtime_ci[0],
                "runtime_ci95_high_seconds": runtime_ci[1],
                "throughput_tokens_mean": _mean(token_throughput),
                "throughput_tokens_median": _median(token_throughput),
                "throughput_tokens_std": _std(token_throughput),
                "throughput_tokens_ci95_low": token_ci[0],
                "throughput_tokens_ci95_high": token_ci[1],
                "throughput_samples_mean": _mean(sample_throughput),
                "throughput_samples_median": _median(sample_throughput),
                "throughput_samples_std": _std(sample_throughput),
                "throughput_samples_ci95_low": sample_ci[0],
                "throughput_samples_ci95_high": sample_ci[1],
            }
        )
    return summaries


def build_strategy_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare throughput-to-memory trade-offs for distributed strategies."""
    selected = _completed_phase(rows, "Distributed training strategy")
    grouped = _group_rows(
        selected,
        "platform_id",
        "comparability",
        "workload",
        "model",
        "precision",
        "strategy",
        "throughput_unit",
        "gpus",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "platform_id": key[0],
                "comparability": key[1],
                "workload": key[2],
                "model": key[3],
                "precision": key[4],
                "strategy": key[5],
                "throughput_unit": key[6],
                "gpus": key[7],
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "strategy",
        "gpus",
        "throughput_unit",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        communication = _mean(_metric_values(group_rows, "allreduce_time_seconds"))
        integrated_communication = _mean(_metric_values(group_rows, "communication_time_seconds"))
        measured = _mean(_metric_values(group_rows, "measured_seconds"))
        summaries.append(
            {
                "condition_id": key[0],
                "experiment_id": key[1],
                "platform_id": key[2],
                "comparability": key[3],
                "workload": key[4],
                "strategy": key[5],
                "gpus": key[6],
                "throughput_unit": key[7],
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
    grouped = _group_rows(
        _completed_phase(rows, "Data movement"),
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "throughput_unit",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summaries.append(
            {
                "condition_id": key[0],
                "experiment_id": key[1],
                "platform_id": key[2],
                "comparability": key[3],
                "workload": key[4],
                "throughput_unit": key[5],
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
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "model",
        "platform",
        "framework",
        "precision",
        "gpus",
        "batch_size",
        "throughput_unit",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summaries.append(
            {
                "condition_id": key[0],
                "experiment_id": key[1],
                "platform_id": key[2],
                "comparability": key[3],
                "workload": key[4],
                "model": key[5],
                "platform": key[6],
                "framework": key[7],
                "precision": key[8],
                "gpus": key[9],
                "batch_size": key[10],
                "throughput_unit": key[11],
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
    grouped = _group_rows(
        _completed_phase(rows, "Memory optimisation"),
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "throughput_unit",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        summaries.append(
            {
                "condition_id": key[0],
                "experiment_id": key[1],
                "platform_id": key[2],
                "comparability": key[3],
                "workload": key[4],
                "throughput_unit": key[5],
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
    grouped = _group_rows(
        selected,
        "condition_id",
        "experiment_id",
        "platform_id",
        "comparability",
        "workload",
        "model",
        "mode",
        "platform",
        "gpu_model",
        "precision",
        "throughput_unit",
    )
    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summaries.append(
            {
                "condition_id": key[0],
                "experiment_id": key[1],
                "platform_id": key[2],
                "comparability": key[3],
                "workload": key[4],
                "model": key[5],
                "mode": key[6],
                "platform": key[7],
                "gpu_model": key[8],
                "precision": key[9],
                "throughput_unit": key[10],
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


def _validate_analysis_identities(rows: list[dict[str, str]]) -> None:
    """Reject processed matrix rows that could enter ambiguous analysis groups."""
    for index, row in enumerate(rows, start=1):
        experiment_id = _value(row, "experiment_id")
        if not is_matrix_experiment_id(experiment_id):
            continue
        platform_id = _value(row, "platform_id")
        condition_id = _value(row, "condition_id")
        comparability = _value(row, "comparability")
        if platform_id not in MATRIX_PLATFORMS:
            raise ValueError(f"processed matrix row {index} has invalid platform_id: {platform_id or '<missing>'}")
        if condition_id != f"{experiment_id}@{platform_id}":
            raise ValueError(f"processed matrix row {index} has invalid condition_id: {condition_id or '<missing>'}")
        if comparability not in COMPARABILITY_VALUES:
            raise ValueError(f"processed matrix row {index} has invalid comparability: {comparability or '<missing>'}")


def _numeric(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _std(values: list[float | None]) -> float | None:
    numeric_values = _numeric(values)
    if not numeric_values:
        return None
    if len(numeric_values) == 1:
        return 0.0
    return statistics.stdev(numeric_values)


def _median(values: list[float | None]) -> float | None:
    numeric_values = _numeric(values)
    if not numeric_values:
        return None
    return statistics.median(numeric_values)


def _ci95(values: list[float | None]) -> tuple[float | None, float | None]:
    """Return a two-sided 95% t interval, or no interval for fewer than two trials."""
    numeric_values = _numeric(values)
    if len(numeric_values) < 2:
        return None, None
    mean = statistics.fmean(numeric_values)
    standard_error = statistics.stdev(numeric_values) / len(numeric_values) ** 0.5
    critical = _t_critical_95(len(numeric_values) - 1)
    margin = critical * standard_error
    return mean - margin, mean + margin


def _t_critical_95(degrees_of_freedom: int) -> float:
    critical_values = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return critical_values.get(degrees_of_freedom, 1.96)


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
