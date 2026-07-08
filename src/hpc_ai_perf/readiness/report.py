from __future__ import annotations

from collections import Counter
from pathlib import Path

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.validation import validate_project
from hpc_ai_perf.workloads import default_registry

WORKLOAD_STATUS = {
    "synthetic": "Ready",
    "environment_validation": "Validation-only",
    "gpt2_small": "Optional smoke-ready",
    "distilbert": "Optional smoke-ready",
    "tinyllama": "Optional inference smoke-ready",
}


def write_readiness_report(
    output_path: str | Path = "results/processed/readiness_report.md",
    *,
    config_root: str | Path = "configs",
    result_dir: str | Path = "results/raw",
    slurm_smoke_dir: str | Path = "slurm/smoke",
) -> Path:
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_readiness_report(config_root=config_root, result_dir=result_dir, slurm_smoke_dir=slurm_smoke_dir),
        encoding="utf-8",
    )
    return report_path


def build_readiness_report(
    *,
    config_root: str | Path = "configs",
    result_dir: str | Path = "results/raw",
    slurm_smoke_dir: str | Path = "slurm/smoke",
) -> str:
    entries = discover_experiment_configs(config_root)
    validation_issues = validate_project(config_root, result_dir)
    workload_counts = Counter(entry.config.workload.name for entry in entries)
    phase_counts = Counter(entry.phase for entry in entries)
    priority_counts = Counter(entry.priority for entry in entries)
    mode_counts = Counter(entry.config.mode for entry in entries)
    registry = default_registry()
    smoke_scripts = sorted(Path(slurm_smoke_dir).glob("*.sbatch"))

    lines = [
        "# Experiment Readiness Report",
        "",
        "Generated from repository configuration, validation checks, workload registry, and Slurm smoke presets.",
        "",
        "## Summary",
        "",
        f"- Matrix configs discovered: {len(entries)}",
        f"- Validation issues: {len(validation_issues)}",
        f"- Registered workloads: {', '.join(registry.names())}",
        f"- Slurm smoke presets: {len(smoke_scripts)}",
        "",
        "## Validation",
        "",
    ]

    if validation_issues:
        lines.extend(["| Severity | Path | Message |", "| --- | --- | --- |"])
        for issue in validation_issues:
            lines.append(f"| {issue.severity} | `{issue.path}` | {issue.message} |")
    else:
        lines.append("Validation passed for configs and current raw result records.")

    lines.extend(
        [
            "",
            "## Workload Coverage",
            "",
            "| Workload | Matrix configs | Registry status |",
            "| --- | ---: | --- |",
        ]
    )
    for workload in sorted(workload_counts):
        lines.append(f"| `{workload}` | {workload_counts[workload]} | {WORKLOAD_STATUS.get(workload, 'Unknown')} |")

    lines.extend(
        [
            "",
            "## Matrix Breakdown",
            "",
            "### By Priority",
            "",
            "| Priority | Count |",
            "| --- | ---: |",
            *_counter_rows(priority_counts),
            "",
            "### By Mode",
            "",
            "| Mode | Count |",
            "| --- | ---: |",
            *_counter_rows(mode_counts),
            "",
            "### By Phase",
            "",
            "| Phase | Count |",
            "| --- | ---: |",
            *_counter_rows(phase_counts),
            "",
            "## Slurm Smoke Presets",
            "",
            "| Script | Present |",
            "| --- | --- |",
        ]
    )
    for script_name in [
        "gpt2_small_smoke.sbatch",
        "distilbert_smoke.sbatch",
        "tinyllama_inference_smoke.sbatch",
    ]:
        present = "yes" if (Path(slurm_smoke_dir) / script_name).exists() else "no"
        lines.append(f"| `slurm/smoke/{script_name}` | {present} |")

    lines.extend(
        [
            "",
            "## Submission Gate",
            "",
            "- Run `python -m hpc_ai_perf.cli validate` before submission.",
            "- Run the three Slurm smoke presets before rendering the full matrix.",
            "- Run `python -m hpc_ai_perf.cli refresh` after smoke results complete.",
            "- Review `results/processed/experiment_tracker.md` and the dashboard before larger batches.",
            "",
        ]
    )
    return "\n".join(lines)


def _counter_rows(counter: Counter[str]) -> list[str]:
    return [f"| {key or 'Unassigned'} | {counter[key]} |" for key in sorted(counter)]
