from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.validation import ValidationIssue, validate_project
from hpc_ai_perf.workloads import default_registry

WORKLOAD_STATUS = {
    "synthetic": "Ready",
    "environment_validation": "Implemented; cluster verification required",
    "gpt2_small": "PyTorch training/inference and vLLM routing implemented",
    "distilbert": "PyTorch training/inference implemented",
    "tinyllama": "PyTorch training/inference and vLLM routing implemented",
}
CORE_ENVIRONMENT_IDS = {"E001", "E002", "E003", "E004", "E006"}
DEEPSPEED_ENVIRONMENT_ID = "E005"
MINIMUM_CORE_TRIALS = 3


def write_readiness_report(
    output_path: str | Path = "results/processed/readiness_report.md",
    *,
    config_root: str | Path = "configs",
    result_dir: str | Path = "results/raw",
    slurm_smoke_dir: str | Path = "slurm/smoke",
) -> Path:
    """Write the repository readiness report to disk.

    The readiness report is a pre-submission snapshot. It combines config
    discovery, validation, workload registry status, and smoke-script presence
    into one Markdown file that can be reviewed before spending HPC GPU budget.
    """
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
    """Build the Markdown readiness report content.

    The report intentionally focuses on matrix readiness rather than local dev
    performance. It answers: how many configs exist, whether validation passes,
    which workloads are registered, whether smoke presets are present, and what
    gates must be cleared before larger Slurm batches.
    """
    entries = discover_experiment_configs(config_root)
    source_entries = [entry for entry in entries if _matrix_number(entry.config.experiment_id) <= 200]
    telemetry_entries = [
        entry for entry in entries if entry.config.metadata.get("template") == "telemetry_revalidation"
    ]
    supplemental_entries = [
        entry
        for entry in entries
        if _matrix_number(entry.config.experiment_id) > 200 and entry not in telemetry_entries
    ]
    validation_issues = validate_project(config_root, result_dir)
    research_issues = validate_project(config_root, result_dir, include_results=False, research_ready=True)
    deferred_entries = [
        entry for entry in entries if entry.config.metadata.get("execution_status", "ready") == "deferred"
    ]
    workload_counts = Counter(entry.config.workload.name for entry in entries)
    phase_counts = Counter(entry.phase for entry in entries)
    priority_counts = Counter(entry.priority for entry in entries)
    mode_counts = Counter(entry.config.mode for entry in entries)
    registry = default_registry()
    smoke_scripts = sorted(Path(slurm_smoke_dir).glob("*.sbatch"))
    valid_completed = _valid_completed_trial_counts(result_dir, validation_issues)
    completed_environment_ids = CORE_ENVIRONMENT_IDS & valid_completed.keys()
    baseline_ids = {
        entry.config.experiment_id
        for entry in entries
        if entry.phase == "Baseline profiling" and entry.priority == "Must Have"
    }
    sufficient_baseline_ids = {
        experiment_id for experiment_id in baseline_ids if valid_completed.get(experiment_id, 0) >= MINIMUM_CORE_TRIALS
    }
    pilot_evidence_ready = completed_environment_ids == CORE_ENVIRONMENT_IDS and sufficient_baseline_ids == baseline_ids

    lines = [
        "# Experiment Readiness Report",
        "",
        "Generated from repository configuration, validation checks, workload registry, and Slurm smoke presets.",
        "",
        "## Summary",
        "",
        f"- Source matrix configs discovered: {len(source_entries)}",
        f"- Supplemental platform probes discovered: {len(supplemental_entries)}",
        f"- Cross-platform telemetry anchors discovered: {len(telemetry_entries)}",
        f"- Total E-number configs discovered: {len(entries)}",
        f"- Validation issues: {len(validation_issues)}",
        f"- Research-readiness blockers: {len(research_issues)}",
        f"- Deferred scope rows: {len(deferred_entries)}",
        f"- Registered workloads: {', '.join(registry.names())}",
        f"- Slurm smoke presets: {len(smoke_scripts)}",
        f"- Core environment checks completed: {len(completed_environment_ids)}/{len(CORE_ENVIRONMENT_IDS)}",
        f"- Baseline definitions with at least {MINIMUM_CORE_TRIALS} trials: "
        f"{len(sufficient_baseline_ids)}/{len(baseline_ids)}",
        f"- Pilot evidence gate: {'READY' if pilot_evidence_ready else 'BLOCKED'}",
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

    lines.extend(["", "## Research Readiness", ""])
    if research_issues:
        lines.extend(["| Severity | Path | Message |", "| --- | --- | --- |"])
        for issue in research_issues:
            lines.append(f"| {issue.severity} | `{issue.path}` | {issue.message} |")
    else:
        lines.append("All non-deferred matrix configs have a semantically valid execution path.")

    lines.extend(["", "## Deferred Scope", ""])
    if deferred_entries:
        lines.extend(
            [
                "These rows remain in the planning matrix but are excluded from deployment batches until the stated "
                "backend or measurement contract is supplied.",
                "",
                "| Experiment | Reason |",
                "| --- | --- |",
            ]
        )
        lines.extend(
            f"| `{entry.config.experiment_id}` | {entry.config.metadata.get('defer_reason', 'No reason recorded')} |"
            for entry in deferred_entries
        )
    else:
        lines.append("No matrix rows are deferred.")

    lines.extend(
        [
            "",
            "## Cluster Evidence Gate",
            "",
            f"- Core environment validations: {_gate_list(CORE_ENVIRONMENT_IDS, valid_completed, 1)}",
            f"- DeepSpeed environment validation: {_gate_list({DEEPSPEED_ENVIRONMENT_ID}, valid_completed, 1)}",
            f"- Must Have baseline repetitions: {_gate_list(baseline_ids, valid_completed, MINIMUM_CORE_TRIALS)}",
            "",
            "The pilot gate remains blocked until every core environment check has one valid completion and every "
            f"Must Have baseline definition has at least {MINIMUM_CORE_TRIALS} valid trials. E005 is an additional "
            "gate before DeepSpeed strategy batches.",
        ]
    )

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
            "- Run `python -m hpc_ai_perf.cli validate --skip-results --research-ready` for the selected batch.",
            "- Run the three Slurm smoke presets before rendering the full matrix.",
            "- Run `python -m hpc_ai_perf.cli refresh` after smoke results complete.",
            "- Review `results/processed/experiment_tracker.md` and the dashboard before larger batches.",
            "",
        ]
    )
    return "\n".join(lines)


def _counter_rows(counter: Counter[str]) -> list[str]:
    """Format sorted counter entries as Markdown table rows."""
    return [f"| {key or 'Unassigned'} | {counter[key]} |" for key in sorted(counter)]


def _matrix_number(experiment_id: str) -> int:
    """Return the numeric part of an E-number experiment identifier."""
    if len(experiment_id) == 4 and experiment_id.startswith("E") and experiment_id[1:].isdigit():
        return int(experiment_id[1:])
    return 10_000


def _valid_completed_trial_counts(result_dir: str | Path, validation_issues: list[ValidationIssue]) -> dict[str, int]:
    invalid_paths = {issue.path for issue in validation_issues}
    counts: dict[str, int] = defaultdict(int)
    for source_path in sorted(Path(result_dir).rglob("*.json")):
        if source_path in invalid_paths:
            continue
        try:
            record = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or record.get("status") != "completed":
            continue
        experiment_id = str(record.get("experiment_id", ""))
        if experiment_id.startswith("E"):
            counts[experiment_id] += 1
    return dict(counts)


def _gate_list(experiment_ids: set[str], counts: dict[str, int], required: int) -> str:
    missing = [
        f"{experiment_id} ({counts.get(experiment_id, 0)}/{required})"
        for experiment_id in sorted(experiment_ids)
        if counts.get(experiment_id, 0) < required
    ]
    return "passed" if not missing else "pending " + ", ".join(missing)
