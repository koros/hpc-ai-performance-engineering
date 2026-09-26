"""Controlled low-bit comparisons; never pool load levels, artifacts or platforms."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from hpc_ai_perf.config.lowbit import METHOD_IDS, PARAMETERS, PROTOCOL, QUALITY_METHOD_IDS, QUALITY_PROTOCOL
from hpc_ai_perf.validation.lowbit import energy_coverage_valid

LOWBIT_COLUMNS = [
    "study_protocol",
    "platform_id",
    "comparability",
    "experiment_id",
    "control_id",
    "method",
    "batch_size",
    "evidence_status",
    "completed_trials",
    "required_trials",
    "mean_throughput_tokens_sec",
    "throughput_ci95_low",
    "throughput_ci95_high",
    "throughput_change_pct",
    "mean_latency_p95_ms",
    "latency_p95_change_pct",
    "mean_ttft_p95_ms",
    "mean_model_memory_gb",
    "model_memory_saving_pct",
    "mean_worker_peak_allocated_gb",
    "mean_quality_perplexity",
    "perplexity_change_pct",
    "quality_tolerance_pct",
    "quality_status",
    "energy_trials",
    "mean_joules_per_output_token",
    "energy_per_token_change_pct",
    "comparison_scope",
    "interpretation",
    "source_paths",
]


def build_lowbit_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_protocol_summary(rows, PROTOCOL, METHOD_IDS) + _build_protocol_summary(
        rows, QUALITY_PROTOCOL, {method: (eid,) for method, eid in QUALITY_METHOD_IDS.items()}
    )


def _build_protocol_summary(rows: list[dict[str, Any]], protocol: str, method_ids: dict) -> list[dict[str, Any]]:
    selected = [r for r in rows if r.get("parameter_study_protocol") == protocol]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[(str(row.get("platform_id")), str(row.get("experiment_id")))].append(row)
    results = []
    for platform in sorted({key[0] for key in groups} & {"l40s", "h100"}):
        for method, ids in method_ids.items():
            for experiment_id, batch in zip(ids, (1, 8)[: len(ids)], strict=True):
                control_id = method_ids["bf16"][0 if batch == 1 else 1]
                own = groups[(platform, experiment_id)]
                controls = groups[(platform, control_id)]
                completed = [r for r in own if r.get("status") == "completed"]
                control_rows = [r for r in controls if r.get("status") == "completed"]
                status = evidence_status(own, method, batch)
                if status == "complete" and evidence_status(controls, "bf16", batch) != "complete":
                    status = "control_incomplete"
                if status == "complete" and comparison_signature(completed[0]) != comparison_signature(control_rows[0]):
                    status = "protocol_mismatch"
                throughput = numbers(completed, "throughput_tokens_sec")
                lower, upper = ci95(throughput)
                energy = energy_per_token(completed)
                control_energy = energy_per_token(control_rows)
                comparison_ok = status == "complete"
                changes = {}
                for label, field in (
                    ("throughput_change_pct", "throughput_tokens_sec"),
                    ("latency_p95_change_pct", "latency_p95_ms"),
                    ("perplexity_change_pct", "metric_quality_perplexity"),
                ):
                    changes[label] = (
                        percent_change(mean(numbers(completed, field)), mean(numbers(control_rows, field)))
                        if comparison_ok
                        else None
                    )
                memory_change = (
                    percent_change(
                        mean(numbers(completed, "metric_memory_model_weights_gb")),
                        mean(numbers(control_rows, "metric_memory_model_weights_gb")),
                    )
                    if comparison_ok
                    else None
                )
                tolerance = (
                    number(completed[0].get("parameter_quality_max_relative_ppl_increase")) if completed else None
                )
                quality = "not_comparable"
                if comparison_ok and changes["perplexity_change_pct"] is not None and tolerance is not None:
                    quality = (
                        "within_tolerance"
                        if changes["perplexity_change_pct"] <= tolerance * 100
                        else "outside_tolerance"
                    )
                if status != "complete":
                    interpretation = f"No comparative conclusion: {status.replace('_', ' ')}."
                elif method == "bf16":
                    interpretation = "Fresh BF16 reference at this load; not pooled with older model trials."
                elif quality == "outside_tolerance":
                    interpretation = (
                        "Text-perplexity degradation exceeds the preregistered tolerance; "
                        "do not recommend on speed alone."
                    )
                else:
                    interpretation = (
                        "Compare observed speed, tail latency and weight-memory savings; "
                        "trial CIs are not a significance test."
                    )
                results.append(
                    {
                        "study_protocol": protocol,
                        "platform_id": platform,
                        "comparability": "exact",
                        "experiment_id": experiment_id,
                        "control_id": control_id,
                        "method": method,
                        "batch_size": batch,
                        "evidence_status": status,
                        "completed_trials": len(completed),
                        "required_trials": 3,
                        "mean_throughput_tokens_sec": mean(throughput),
                        "throughput_ci95_low": lower,
                        "throughput_ci95_high": upper,
                        **changes,
                        "mean_latency_p95_ms": mean(numbers(completed, "latency_p95_ms")),
                        "mean_ttft_p95_ms": mean(numbers(completed, "metric_time_to_first_token_p95_ms")),
                        "mean_model_memory_gb": mean(numbers(completed, "metric_memory_model_weights_gb")),
                        "model_memory_saving_pct": -memory_change if memory_change is not None else None,
                        "mean_worker_peak_allocated_gb": mean(
                            numbers(completed, "metric_memory_worker_peak_allocated_gb")
                        ),
                        "mean_quality_perplexity": mean(numbers(completed, "metric_quality_perplexity")),
                        "quality_tolerance_pct": tolerance * 100 if tolerance is not None else None,
                        "quality_status": quality,
                        "energy_trials": len(energy),
                        "mean_joules_per_output_token": mean(energy),
                        "energy_per_token_change_pct": percent_change(mean(energy), mean(control_energy))
                        if comparison_ok and len(energy) == len(control_energy) == 3
                        else None,
                        "comparison_scope": "publisher_artifact_deployment;not_pure_quantizer_effect",
                        "interpretation": interpretation,
                        "source_paths": ";".join(str(r.get("source_path", "")) for r in own),
                    }
                )
    return results


def comparison_signature(row: dict[str, Any]) -> tuple[str, ...]:
    fields = [f"parameter_{p}" for p in sorted(PARAMETERS - {"quantization_method", "comparison_control_id"})]
    fields += [
        "platform_id",
        "comparability",
        "gpus",
        "throughput_unit",
        "metric_study_runtime_sha256",
        "metric_runtime_packages",
        "metric_performance_tokens_sha256",
        "metric_quality_tokens_sha256",
        "metric_base_model_revision",
        "metric_vllm_execution_mode",
    ]
    return tuple(str(row.get(field, "")) for field in fields)


def evidence_status(rows: list[dict[str, Any]], method: str, batch: int) -> str:
    completed = [r for r in rows if r.get("status") == "completed"]
    if not rows:
        return "not_run"
    repetitions = [str(r.get("repetition_number")) for r in completed]
    if len(set(repetitions)) != len(repetitions) or any(r not in {"1", "2", "3"} for r in repetitions):
        return "ambiguous_repetitions"
    if len(completed) != 3:
        return "incomplete"
    required = (
        "metric_study_runtime_sha256",
        "metric_runtime_packages",
        "metric_performance_tokens_sha256",
        "metric_quality_tokens_sha256",
        "metric_base_model_revision",
        "metric_vllm_execution_mode",
    )
    for row in completed:
        if any(not row.get(key) for key in required):
            return "missing_protocol_evidence"
        if (
            row.get("parameter_quantization_method") != method
            or row.get("metric_quantization_method") != method
            or number(row.get("batch_size")) != batch
            or row.get("comparability") != "exact"
            or number(row.get("gpus")) != 1
            or row.get("throughput_unit") != "generated_tokens_per_second"
        ):
            return "protocol_mismatch"
        if any(
            number(row.get(key)) is None or number(row.get(key)) <= 0
            for key in (
                "throughput_tokens_sec",
                "metric_quality_perplexity",
                "latency_p95_ms",
                "metric_memory_model_weights_gb",
                "metric_time_to_first_token_p95_ms",
            )
        ):
            return "invalid_metrics"
    if (
        len({comparison_signature(row) for row in completed}) != 1
        or len({row.get("config_sha256") for row in completed}) != 1
    ):
        return "protocol_mismatch"
    return "complete"


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def numbers(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [value for row in rows if (value := number(row.get(field))) is not None]


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def percent_change(value: float | None, reference: float | None) -> float | None:
    return (value / reference - 1) * 100 if value is not None and reference is not None and reference > 0 else None


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) not in {2, 3}:
        return None, None
    average = statistics.mean(values)
    # Student-t, two-sided 95%, for two or three independent process runs.
    radius = {2: 12.706, 3: 4.303}[len(values)] * statistics.stdev(values) / math.sqrt(len(values))
    return average - radius, average + radius


def energy_per_token(rows: list[dict[str, Any]]) -> list[float]:
    values = []
    for row in rows:
        metrics = {
            "energy_scope": row.get("energy_scope"),
            "energy_measurement_status": row.get("energy_measurement_status"),
            "nvidia_smi.gpu_mapping_method": row.get("metric_nvidia_smi_gpu_mapping_method"),
        }
        for name in (
            "energy_joules_measured_region",
            "nvidia_smi.measurement_interior_samples",
            "nvidia_smi.measurement_max_gap_seconds",
        ):
            metrics[name] = number(row.get("metric_" + name.replace(".", "_")))
        tokens = number(row.get("metric_generated_tokens"))
        if energy_coverage_valid(metrics) and tokens is not None and tokens > 0:
            values.append(metrics["energy_joules_measured_region"] / tokens)
    return values
