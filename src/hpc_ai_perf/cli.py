from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from hpc_ai_perf.analysis import analyze_results
from hpc_ai_perf.config import (
    discover_experiment_configs,
    entries_to_json,
    entries_to_table,
    filter_config_entries,
    load_experiment_config,
)
from hpc_ai_perf.readiness import write_readiness_report
from hpc_ai_perf.results import aggregate_results
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.slurm import (
    record_job_submission,
    render_slurm_script,
    write_job_status_csv,
    write_job_status_report,
    write_slurm_script,
    write_submission_plan,
)
from hpc_ai_perf.tracking import track_experiment_progress
from hpc_ai_perf.validation import format_validation_issues, validate_project

LOGGER = logging.getLogger(__name__)
SLURM_PROFILE_KEYS = {
    "account",
    "cpus_per_task",
    "environment",
    "modules",
    "partition",
    "setup_commands",
    "time",
}
SLURM_PROFILE_PLACEHOLDERS = ("replace-with", "TODO", "changeme", "<")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list-configs":
        entries = discover_experiment_configs(args.config_root, matrix_only=not args.include_templates)
        entries = filter_config_entries(
            entries,
            phase=args.phase,
            priority=args.priority,
            mode=args.mode,
            workload=args.workload,
            experiment_id=args.experiment_id,
        )
        if args.format == "json":
            print(entries_to_json(entries), end="")
        else:
            print(entries_to_table(entries), end="")
        return

    if args.command == "run":
        _run_local_config(args.config, args.output_dir)
        return

    if args.command == "render-slurm":
        _render_slurm_config(
            args.config,
            output_dir=args.output_dir,
            partition=args.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
            profile_path=args.slurm_profile,
            print_script=args.print,
        )
        return

    if args.command == "render-slurm-bulk":
        _render_slurm_bulk(
            config_root=args.config_root,
            output_dir=args.output_dir,
            phase=args.phase,
            priority=args.priority,
            mode=args.mode,
            workload=args.workload,
            experiment_id=args.experiment_id,
            partition=args.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
            profile_path=args.slurm_profile,
        )
        return

    if args.command == "plan-slurm-submission":
        plan_path = write_submission_plan(
            args.script_dir,
            args.output,
            batch_name=args.batch_name,
        )
        LOGGER.info("Wrote Slurm submission plan to %s", plan_path)
        return

    if args.command == "init-slurm-status":
        status_path = write_job_status_csv(
            args.script_dir,
            args.output,
            batch_name=args.batch_name,
        )
        LOGGER.info("Wrote Slurm job status CSV to %s", status_path)
        return

    if args.command == "record-slurm-job":
        status_path = record_job_submission(
            args.status,
            experiment_id=args.experiment_id,
            script_path=args.script_path,
            job_id=args.job_id,
            sbatch_output=args.sbatch_output,
            state=args.state,
            submitted_at=args.submitted_at,
            notes=args.notes,
        )
        LOGGER.info("Updated Slurm job status CSV at %s", status_path)
        return

    if args.command == "report-slurm-status":
        report_path = write_job_status_report(args.status, args.output)
        LOGGER.info("Wrote Slurm job status report to %s", report_path)
        return

    if args.command == "check-slurm-profile":
        issues = _check_slurm_profile(args.slurm_profile)
        if issues:
            print("Slurm profile is not ready:\n", end="")
            for issue in issues:
                print(f"- {issue}")
            raise SystemExit(1)
        print(f"Slurm profile is ready: {args.slurm_profile}")
        return

    if args.command == "aggregate-results":
        result_path = aggregate_results(args.input_dir, args.output)
        LOGGER.info("Wrote aggregated results to %s", result_path)
        return

    if args.command == "analyze-results":
        written_paths = analyze_results(args.input, args.output_dir)
        LOGGER.info("Wrote %d analysis tables to %s", len(written_paths), args.output_dir)
        return

    if args.command == "track-progress":
        csv_path, markdown_path = track_experiment_progress(
            args.config_root,
            args.result_dir,
            args.output_csv,
            args.output_markdown,
        )
        LOGGER.info("Wrote tracker CSV to %s", csv_path)
        LOGGER.info("Wrote tracker Markdown to %s", markdown_path)
        return

    if args.command == "validate":
        issues = validate_project(args.config_root, args.result_dir, include_results=not args.skip_results)
        if args.format == "json":
            print(json.dumps([issue.to_dict() for issue in issues], indent=2) + "\n", end="")
        else:
            print(format_validation_issues(issues), end="")
        if issues:
            raise SystemExit(1)
        return

    if args.command == "refresh":
        _refresh_outputs(
            config_root=args.config_root,
            result_dir=args.result_dir,
            processed_results=args.processed_results,
            analysis_dir=args.analysis_dir,
            tracker_csv=args.tracker_csv,
            tracker_markdown=args.tracker_markdown,
            readiness_report=args.readiness_report,
            skip_validation=args.skip_validation,
        )
        return

    if args.command == "readiness-report":
        report_path = write_readiness_report(
            args.output,
            config_root=args.config_root,
            result_dir=args.result_dir,
            slurm_smoke_dir=args.slurm_smoke_dir,
        )
        LOGGER.info("Wrote readiness report to %s", report_path)
        return

    parser.error("A command is required.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage and run HPC AI benchmark experiments.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment config locally.")
    run_parser.add_argument("--config", required=True, help="Path to a JSON or YAML experiment config.")
    run_parser.add_argument("--output-dir", default=None, help="Override the configured output directory.")

    list_parser = subparsers.add_parser("list-configs", help="List available experiment configs.")
    list_parser.add_argument("--config-root", default="configs", help="Root directory containing config files.")
    list_parser.add_argument("--include-templates", action="store_true", help="Include non-matrix template configs.")
    list_parser.add_argument("--phase", default=None, help="Filter by metadata phase.")
    list_parser.add_argument("--priority", default=None, help="Filter by metadata priority.")
    list_parser.add_argument("--mode", choices=["training", "inference", "profiling"], default=None)
    list_parser.add_argument("--workload", default=None, help="Filter by workload name.")
    list_parser.add_argument("--experiment-id", default=None, help="Filter by experiment ID, such as E001.")
    list_parser.add_argument("--format", choices=["table", "json"], default="table")

    slurm_parser = subparsers.add_parser("render-slurm", help="Render an sbatch script for an experiment config.")
    slurm_parser.add_argument("--config", required=True, help="Path to a JSON or YAML experiment config.")
    _add_slurm_options(slurm_parser)
    slurm_parser.add_argument("--print", action="store_true", help="Print rendered script instead of writing it.")

    bulk_parser = subparsers.add_parser("render-slurm-bulk", help="Render sbatch scripts for filtered configs.")
    bulk_parser.add_argument("--config-root", default="configs", help="Root directory containing config files.")
    bulk_parser.add_argument("--phase", default=None, help="Filter by metadata phase.")
    bulk_parser.add_argument("--priority", default=None, help="Filter by metadata priority.")
    bulk_parser.add_argument("--mode", choices=["training", "inference", "profiling"], default=None)
    bulk_parser.add_argument("--workload", default=None, help="Filter by workload name.")
    bulk_parser.add_argument("--experiment-id", default=None, help="Filter by experiment ID, such as E001.")
    _add_slurm_options(bulk_parser)

    plan_slurm_parser = subparsers.add_parser(
        "plan-slurm-submission",
        help="Write a reviewable submission plan for generated sbatch scripts.",
    )
    plan_slurm_parser.add_argument("--script-dir", required=True, help="Directory containing generated sbatch scripts.")
    plan_slurm_parser.add_argument("--output", required=True, help="Path for the generated Markdown submission plan.")
    plan_slurm_parser.add_argument("--batch-name", default=None, help="Optional display name for the batch.")

    init_status_parser = subparsers.add_parser(
        "init-slurm-status",
        help="Initialize a CSV for tracking generated sbatch scripts through Slurm submission.",
    )
    init_status_parser.add_argument(
        "--script-dir", required=True, help="Directory containing generated sbatch scripts."
    )
    init_status_parser.add_argument("--output", required=True, help="Path for the generated status CSV.")
    init_status_parser.add_argument("--batch-name", default=None, help="Optional display name for the batch.")

    record_status_parser = subparsers.add_parser(
        "record-slurm-job",
        help="Record a Slurm job ID against an experiment in a status CSV.",
    )
    record_status_parser.add_argument("--status", required=True, help="Path to a Slurm job status CSV.")
    record_status_parser.add_argument("--experiment-id", default=None, help="Experiment ID to update, such as E001.")
    record_status_parser.add_argument("--script-path", default=None, help="Script path to update.")
    record_status_parser.add_argument("--job-id", default=None, help="Slurm job ID.")
    record_status_parser.add_argument(
        "--sbatch-output",
        default=None,
        help="Raw sbatch output, for example 'Submitted batch job 12345'.",
    )
    record_status_parser.add_argument("--state", default="SUBMITTED", help="State to record for the job.")
    record_status_parser.add_argument("--submitted-at", default=None, help="Optional submission timestamp.")
    record_status_parser.add_argument("--notes", default=None, help="Optional notes for the status row.")

    report_status_parser = subparsers.add_parser(
        "report-slurm-status",
        help="Write a Markdown report from a Slurm job status CSV.",
    )
    report_status_parser.add_argument("--status", required=True, help="Path to a Slurm job status CSV.")
    report_status_parser.add_argument("--output", required=True, help="Path for the generated Markdown report.")

    check_profile_parser = subparsers.add_parser(
        "check-slurm-profile",
        help="Validate that a Slurm profile is ready for real cluster rendering.",
    )
    check_profile_parser.add_argument("--slurm-profile", required=True, help="Path to a Slurm profile JSON file.")

    aggregate_parser = subparsers.add_parser("aggregate-results", help="Aggregate raw result JSON files into CSV.")
    aggregate_parser.add_argument(
        "--input-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    aggregate_parser.add_argument(
        "--output",
        default="results/processed/experiments.csv",
        help="Path for the processed experiment CSV.",
    )

    analyze_parser = subparsers.add_parser("analyze-results", help="Build analysis tables from processed results.")
    analyze_parser.add_argument(
        "--input",
        default="results/processed/experiments.csv",
        help="Processed experiment CSV created by aggregate-results.",
    )
    analyze_parser.add_argument("--output-dir", default="results/analysis", help="Directory for analysis CSV tables.")

    tracker_parser = subparsers.add_parser("track-progress", help="Build tracker reports from configs and raw results.")
    tracker_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    tracker_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    tracker_parser.add_argument(
        "--output-csv",
        default="results/processed/experiment_tracker.csv",
        help="Path for the generated tracker CSV.",
    )
    tracker_parser.add_argument(
        "--output-markdown",
        default="results/processed/experiment_tracker.md",
        help="Path for the generated tracker Markdown.",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate experiment configs and raw result records.")
    validate_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    validate_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    validate_parser.add_argument("--skip-results", action="store_true", help="Only validate experiment configs.")
    validate_parser.add_argument("--format", choices=["table", "json"], default="table")

    refresh_parser = subparsers.add_parser("refresh", help="Validate and rebuild processed project outputs.")
    refresh_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    refresh_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    refresh_parser.add_argument(
        "--processed-results",
        default="results/processed/experiments.csv",
        help="Path for the aggregated experiment CSV.",
    )
    refresh_parser.add_argument("--analysis-dir", default="results/analysis", help="Directory for analysis CSV tables.")
    refresh_parser.add_argument(
        "--tracker-csv",
        default="results/processed/experiment_tracker.csv",
        help="Path for the generated tracker CSV.",
    )
    refresh_parser.add_argument(
        "--tracker-markdown",
        default="results/processed/experiment_tracker.md",
        help="Path for the generated tracker Markdown.",
    )
    refresh_parser.add_argument(
        "--readiness-report",
        default="results/processed/readiness_report.md",
        help="Path for the generated readiness report.",
    )
    refresh_parser.add_argument("--skip-validation", action="store_true", help="Rebuild outputs without validation.")

    readiness_parser = subparsers.add_parser("readiness-report", help="Write an HPC submission readiness report.")
    readiness_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    readiness_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    readiness_parser.add_argument(
        "--slurm-smoke-dir",
        default="slurm/smoke",
        help="Directory containing Slurm smoke preset scripts.",
    )
    readiness_parser.add_argument(
        "--output",
        default="results/processed/readiness_report.md",
        help="Path for the generated readiness report.",
    )
    return parser


def _add_slurm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="slurm/generated", help="Directory for generated sbatch scripts.")
    parser.add_argument("--slurm-profile", default=None, help="Optional JSON file with reusable Slurm defaults.")
    parser.add_argument("--partition", default=None, help="Optional Slurm partition.")
    parser.add_argument("--account", default=None, help="Optional Slurm account.")
    parser.add_argument("--time", default=None, help="Override Slurm time limit, for example 02:00:00.")
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--module", action="append", default=[], help="Module to load; can be repeated.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable KEY=VALUE; can be repeated.",
    )


def _run_local_config(config_path: str, output_dir: str | None) -> None:
    config = load_experiment_config(config_path)
    runner = LocalBenchmarkRunner()
    result = runner.run(config)
    result_path = result.write_json(output_dir or config.runtime.output_dir)
    LOGGER.info("Wrote result to %s", result_path)


def _refresh_outputs(
    *,
    config_root: str,
    result_dir: str,
    processed_results: str,
    analysis_dir: str,
    tracker_csv: str,
    tracker_markdown: str,
    readiness_report: str,
    skip_validation: bool,
) -> None:
    if not skip_validation:
        issues = validate_project(config_root, result_dir)
        if issues:
            print(format_validation_issues(issues), end="")
            raise SystemExit(1)
        LOGGER.info("Validation passed.")

    result_path = aggregate_results(result_dir, processed_results)
    LOGGER.info("Wrote aggregated results to %s", result_path)

    analysis_paths = analyze_results(result_path, analysis_dir)
    LOGGER.info("Wrote %d analysis tables to %s", len(analysis_paths), analysis_dir)

    tracker_csv_path, tracker_markdown_path = track_experiment_progress(
        config_root,
        result_dir,
        tracker_csv,
        tracker_markdown,
    )
    LOGGER.info("Wrote tracker CSV to %s", tracker_csv_path)
    LOGGER.info("Wrote tracker Markdown to %s", tracker_markdown_path)

    readiness_path = write_readiness_report(
        readiness_report,
        config_root=config_root,
        result_dir=result_dir,
    )
    LOGGER.info("Wrote readiness report to %s", readiness_path)


def _render_slurm_config(
    config_path: str,
    *,
    output_dir: str,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
    profile_path: str | None,
    print_script: bool,
) -> None:
    slurm_options = _resolve_slurm_options(
        profile_path=profile_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        env_values=env_values,
    )
    config = load_experiment_config(config_path)
    script = render_slurm_script(
        config,
        config_path,
        partition=slurm_options["partition"],
        account=slurm_options["account"],
        time_limit=slurm_options["time_limit"],
        cpus_per_task=slurm_options["cpus_per_task"],
        modules=slurm_options["modules"],
        environment=slurm_options["environment"],
        setup_commands=slurm_options["setup_commands"],
    )
    if print_script:
        print(script, end="")
        return

    script_path = write_slurm_script(script, output_dir, config.experiment_id)
    LOGGER.info("Wrote Slurm script to %s", script_path)


def _render_slurm_bulk(
    *,
    config_root: str,
    output_dir: str,
    phase: str | None,
    priority: str | None,
    mode: str | None,
    workload: str | None,
    experiment_id: str | None,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
    profile_path: str | None,
) -> None:
    entries = discover_experiment_configs(config_root)
    entries = filter_config_entries(
        entries,
        phase=phase,
        priority=priority,
        mode=mode,
        workload=workload,
        experiment_id=experiment_id,
    )
    if not entries:
        LOGGER.warning("No configs matched the requested filters.")
        return

    slurm_options = _resolve_slurm_options(
        profile_path=profile_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        env_values=env_values,
    )
    for entry in entries:
        script = render_slurm_script(
            entry.config,
            entry.path,
            partition=slurm_options["partition"],
            account=slurm_options["account"],
            time_limit=slurm_options["time_limit"],
            cpus_per_task=slurm_options["cpus_per_task"],
            modules=slurm_options["modules"],
            environment=slurm_options["environment"],
            setup_commands=slurm_options["setup_commands"],
        )
        write_slurm_script(script, output_dir, entry.config.experiment_id)

    LOGGER.info("Wrote %d Slurm scripts to %s", len(entries), output_dir)


def _parse_environment(values: list[str]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected environment value in KEY=VALUE form: {value}")
        key, env_value = value.split("=", 1)
        if not key:
            raise ValueError(f"Environment key cannot be empty: {value}")
        environment[key] = env_value
    return environment


def _resolve_slurm_options(
    *,
    profile_path: str | None,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
) -> dict[str, Any]:
    profile = _load_slurm_profile(profile_path) if profile_path else {}
    profile_environment = profile.get("environment", {})

    return {
        "partition": partition or profile.get("partition"),
        "account": account or profile.get("account"),
        "time_limit": time_limit or profile.get("time"),
        "cpus_per_task": cpus_per_task or profile.get("cpus_per_task", 8),
        "modules": [*profile.get("modules", []), *modules],
        "environment": {**profile_environment, **_parse_environment(env_values)},
        "setup_commands": profile.get("setup_commands", []),
    }


def _load_slurm_profile(profile_path: str) -> dict[str, Any]:
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"Slurm profile must be a JSON object: {profile_path}")

    unknown_keys = sorted(set(profile) - SLURM_PROFILE_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown Slurm profile keys in {profile_path}: {', '.join(unknown_keys)}")

    _validate_string_or_none(profile, "partition", profile_path)
    _validate_string_or_none(profile, "account", profile_path)
    _validate_string_or_none(profile, "time", profile_path)
    _validate_int_or_none(profile, "cpus_per_task", profile_path)
    _validate_string_list(profile, "modules", profile_path)
    _validate_string_list(profile, "setup_commands", profile_path)
    _validate_string_dict(profile, "environment", profile_path)
    return profile


def _check_slurm_profile(profile_path: str) -> list[str]:
    profile = _load_slurm_profile(profile_path)
    issues: list[str] = []

    for key in ("partition", "account"):
        value = profile.get(key)
        if not value:
            issues.append(f"`{key}` must be set.")
        elif _contains_placeholder(value):
            issues.append(f"`{key}` still contains a placeholder: {value}")

    modules = profile.get("modules", [])
    if not modules:
        issues.append("`modules` should list the CUDA/Python stack required on the cluster.")

    setup_commands = profile.get("setup_commands", [])
    for index, command in enumerate(setup_commands, start=1):
        if _contains_placeholder(command):
            issues.append(f"`setup_commands[{index}]` still contains a placeholder: {command}")

    environment = profile.get("environment", {})
    for key, value in environment.items():
        if _contains_placeholder(value):
            issues.append(f"`environment.{key}` still contains a placeholder: {value}")

    return issues


def _contains_placeholder(value: str) -> bool:
    return any(token.lower() in value.lower() for token in SLURM_PROFILE_PLACEHOLDERS)


def _validate_string_or_none(profile: dict[str, Any], key: str, profile_path: str) -> None:
    if key in profile and profile[key] is not None and not isinstance(profile[key], str):
        raise ValueError(f"Slurm profile key '{key}' must be a string: {profile_path}")


def _validate_int_or_none(profile: dict[str, Any], key: str, profile_path: str) -> None:
    if key in profile and profile[key] is not None and not isinstance(profile[key], int):
        raise ValueError(f"Slurm profile key '{key}' must be an integer: {profile_path}")


def _validate_string_list(profile: dict[str, Any], key: str, profile_path: str) -> None:
    if key not in profile:
        return
    value = profile[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Slurm profile key '{key}' must be a list of strings: {profile_path}")


def _validate_string_dict(profile: dict[str, Any], key: str, profile_path: str) -> None:
    if key not in profile:
        return
    value = profile[key]
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"Slurm profile key '{key}' must be an object of string values: {profile_path}")


if __name__ == "__main__":
    main()
