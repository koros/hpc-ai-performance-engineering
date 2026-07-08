from __future__ import annotations

import argparse
import logging

from hpc_ai_perf.config import (
    discover_experiment_configs,
    entries_to_json,
    entries_to_table,
    filter_config_entries,
    load_experiment_config,
)
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.slurm import render_slurm_script, write_slurm_script

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
    slurm_parser.add_argument("--output-dir", default="slurm/generated", help="Directory for generated sbatch scripts.")
    slurm_parser.add_argument("--partition", default=None, help="Optional Slurm partition.")
    slurm_parser.add_argument("--account", default=None, help="Optional Slurm account.")
    slurm_parser.add_argument("--time", default=None, help="Override Slurm time limit, for example 02:00:00.")
    slurm_parser.add_argument("--cpus-per-task", type=int, default=8)
    slurm_parser.add_argument("--module", action="append", default=[], help="Module to load; can be repeated.")
    slurm_parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable KEY=VALUE; can be repeated.",
    )
    slurm_parser.add_argument("--print", action="store_true", help="Print rendered script instead of writing it.")
    return parser


def _run_local_config(config_path: str, output_dir: str | None) -> None:
    config = load_experiment_config(config_path)
    runner = LocalBenchmarkRunner()
    result = runner.run(config)
    result_path = result.write_json(output_dir or config.runtime.output_dir)
    LOGGER.info("Wrote result to %s", result_path)


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
