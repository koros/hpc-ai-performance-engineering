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
    return parser


def _run_local_config(config_path: str, output_dir: str | None) -> None:
    config = load_experiment_config(config_path)
    runner = LocalBenchmarkRunner()
    result = runner.run(config)
    result_path = result.write_json(output_dir or config.runtime.output_dir)
    LOGGER.info("Wrote result to %s", result_path)


if __name__ == "__main__":
    main()
