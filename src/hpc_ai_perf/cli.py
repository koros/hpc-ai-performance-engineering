from __future__ import annotations

import argparse
import logging

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.runners import LocalBenchmarkRunner

LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run an HPC AI benchmark experiment locally.")
    parser.add_argument("--config", required=True, help="Path to a JSON or YAML experiment config.")
    parser.add_argument("--output-dir", default=None, help="Override the configured output directory.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    runner = LocalBenchmarkRunner()
    result = runner.run(config)
    output_dir = args.output_dir or config.runtime.output_dir
    result_path = result.write_json(output_dir)
    LOGGER.info("Wrote result to %s", result_path)


if __name__ == "__main__":
    main()
