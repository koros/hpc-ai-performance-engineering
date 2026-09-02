from __future__ import annotations

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hpc_ai_perf.result_identity import is_matrix_experiment_id, resolve_matrix_result_identity
from hpc_ai_perf.results.aggregate import load_result_records, telemetry_mapping_class

TELEMETRY_METRICS = {
    "gpu_utilization": ("nvidia_smi.gpu_utilization", 0.50),
    "memory_used_gb": ("nvidia_smi.memory_used_gb", 0.25),
    "power_draw_watts": ("nvidia_smi.power_draw_watts", 0.25),
}
AUDIT_COLUMNS = [
    "condition_id",
    "experiment_id",
    "platform_id",
    "repetition_number",
    "metric",
    "value",
    "group_median",
    "relative_deviation",
    "issue",
    "gpu_telemetry_mapping",
    "source_path",
]


def audit_telemetry(
    result_root: str | Path = "results/raw",
    csv_output: str | Path = "results/analysis/telemetry_audit.csv",
    markdown_output: str | Path = "results/analysis/telemetry_audit.md",
) -> tuple[Path, Path]:
    """Audit immutable raw evidence for telemetry provenance and repeated-trial outliers."""
    records = load_result_records(result_root)
    mapping_counts: Counter[str] = Counter()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for source_path, record in records:
        metrics = _mapping(record.get("metrics"))
        mapping = telemetry_mapping_class(metrics)
        mapping_counts[mapping] += 1
        if record.get("status") != "completed":
            continue
        condition_id, platform_id = _identity(record)
        for metric, (raw_name, threshold) in TELEMETRY_METRICS.items():
            value = _number(metrics.get(raw_name))
            if value is None:
                continue
            grouped[(condition_id, metric)].append(
                {
                    "condition_id": condition_id,
                    "experiment_id": str(record.get("experiment_id") or ""),
                    "platform_id": platform_id,
                    "repetition_number": record.get("repetition_number"),
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "gpu_telemetry_mapping": mapping,
                    "source_path": str(source_path),
                }
            )

    issues: list[dict[str, Any]] = []
    audited_groups = 0
    for trials in grouped.values():
        if len(trials) < 3:
            continue
        audited_groups += 1
        median = statistics.median(trial["value"] for trial in trials)
        mapping_values = {trial["gpu_telemetry_mapping"] for trial in trials}
        if len(mapping_values) > 1:
            for trial in trials:
                issues.append(_issue_row(trial, median, None, "mixed_mapping_provenance"))
        for trial in trials:
            physical_issue = _physical_issue(trial["metric"], trial["value"])
            if physical_issue:
                issues.append(_issue_row(trial, median, None, physical_issue))
                continue
            deviation = _relative_deviation(trial["value"], median)
            if deviation is not None and deviation > trial["threshold"]:
                issues.append(_issue_row(trial, median, deviation, "repeated_trial_outlier"))

    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in AUDIT_COLUMNS} for row in issues)

    markdown_path = Path(markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        _render_markdown(
            evidence_records=len(records),
            mapping_counts=mapping_counts,
            audited_groups=audited_groups,
            issues=issues,
        ),
        encoding="utf-8",
    )
    return csv_path, markdown_path


def _identity(record: dict[str, Any]) -> tuple[str, str]:
    experiment_id = str(record.get("experiment_id") or "")
    if is_matrix_experiment_id(experiment_id):
        identity = resolve_matrix_result_identity(record)
        return identity.condition_id, identity.platform_id
    return str(record.get("condition_id") or experiment_id), str(record.get("platform_id") or "unknown")


def _issue_row(trial: dict[str, Any], median: float, deviation: float | None, issue: str) -> dict[str, Any]:
    return {
        **trial,
        "group_median": median,
        "relative_deviation": deviation,
        "issue": issue,
    }


def _physical_issue(metric: str, value: float) -> str | None:
    if metric == "gpu_utilization" and not 0 <= value <= 100:
        return "outside_physical_range"
    if metric in {"memory_used_gb", "power_draw_watts"} and value < 0:
        return "outside_physical_range"
    return None


def _relative_deviation(value: float, median: float) -> float | None:
    if median == 0:
        return None if value == 0 else float("inf")
    return abs(value - median) / abs(median)


def _render_markdown(
    *, evidence_records: int, mapping_counts: Counter[str], audited_groups: int, issues: list[dict[str, Any]]
) -> str:
    lines = [
        "# GPU Telemetry Audit",
        "",
        "Raw evidence is immutable. This report labels telemetry provenance and flags review candidates; it does not "
        "rewrite or automatically exclude any trial.",
        "",
        f"- Evidence records scanned: {evidence_records}",
        f"- Repeated condition/metric groups audited: {audited_groups}",
        f"- Review flags: {len(issues)}",
        "",
        "## Mapping provenance",
        "",
        "| Mapping | Records |",
        "| --- | ---: |",
    ]
    for mapping in ["uuid_verified", "cgroup_verified", "legacy_positional", "not_collected"]:
        lines.append(f"| {mapping} | {mapping_counts.get(mapping, 0)} |")
    lines.extend(
        [
            "",
            "`legacy_positional` means the historical collector sampled the first visible `nvidia-smi` rows without "
            "recording an independent CUDA-to-GPU UUID match. Preserve those trials, but qualify metric-dependent "
            "claims until representative UUID-verified anchors agree within expected variance.",
            "",
            "## Review flags",
            "",
        ]
    )
    if not issues:
        lines.append("No repeated-trial telemetry outliers were flagged by the configured audit thresholds.")
    else:
        lines.extend(
            [
                "| Condition | Trial | Metric | Value | Median | Relative deviation | Issue | Mapping |",
                "| --- | ---: | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for issue in issues:
            deviation = issue.get("relative_deviation")
            formatted_deviation = "-" if deviation is None else f"{float(deviation):.1%}"
            lines.append(
                f"| {issue['condition_id']} | {issue['repetition_number']} | {issue['metric']} | "
                f"{issue['value']:.4g} | {issue['group_median']:.4g} | {formatted_deviation} | "
                f"{issue['issue']} | {issue['gpu_telemetry_mapping']} |"
            )
    return "\n".join(lines) + "\n"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
