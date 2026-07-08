from __future__ import annotations

import argparse
import json
import logging

from hpc_ai_perf.analysis import analyze_results
from hpc_ai_perf.config import (
    discover_experiment_configs,
    entries_to_json,
    entries_to_table,
    filter_config_entries,
    load_experiment_config,
)
from hpc_ai_perf.results import aggregate_results
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.slurm import render_slurm_script, write_slurm_script
from hpc_ai_perf.tracking import track_experiment_progress
from hpc_ai_perf.validation import format_validation_issues, validate_project

LOGGER = logging.getLogger(__name__)


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
        )
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
            skip_validation=args.skip_validation,
        )
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
    refresh_parser.add_argument("--skip-validation", action="store_true", help="Rebuild outputs without validation.")
    return parser


def _add_slurm_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default="slurm/generated", help="Directory for generated sbatch scripts.")
    parser.add_argument("--partition", default=None, help="Optional Slurm partition.")
    parser.add_argument("--account", default=None, help="Optional Slurm account.")
    parser.add_argument("--time", default=None, help="Override Slurm time limit, for example 02:00:00.")
    parser.add_argument("--cpus-per-task", type=int, default=8)
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


def _render_slurm_config(
    config_path: str,
    *,
    output_dir: str,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int,
    modules: list[str],
    env_values: list[str],
    print_script: bool,
) -> None:
    config = load_experiment_config(config_path)
    script = render_slurm_script(
        config,
        config_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        environment=_parse_environment(env_values),
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
    cpus_per_task: int,
    modules: list[str],
    env_values: list[str],
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

    environment = _parse_environment(env_values)
    for entry in entries:
        script = render_slurm_script(
            entry.config,
            entry.path,
            partition=partition,
            account=account,
            time_limit=time_limit,
            cpus_per_task=cpus_per_task,
            modules=modules,
            environment=environment,
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


if __name__ == "__main__":
    main()
