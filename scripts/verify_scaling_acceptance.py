#!/usr/bin/env python3
"""Verify collected TinyLlama scaling canaries before the batch crossover wave."""

import argparse
import json
from pathlib import Path

from hpc_ai_perf.validation.scaling_acceptance import verify_scaling_canaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("h100", "l40s"))
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    result = verify_scaling_canaries(args.result_root, args.platform)
    print(json.dumps({"status": "completed", "platform": args.platform, "results": result}, indent=2))


if __name__ == "__main__":
    main()
