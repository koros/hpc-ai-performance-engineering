#!/usr/bin/env python3
"""Fail closed unless the H100 Qwen2.5-7B sharding canaries are valid.

The canaries are canonical raw evidence after collection, but they are not
performance-ranking data.  Their sole purpose is to establish that the exact
model, BF16 load path, activation checkpointing, and distributed runtime work
before submitting the costlier three-trial strategy study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc_ai_perf.validation.qwen7b_training import verify_canaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--platform", choices=("h100",), default="h100")
    args = parser.parse_args()

    accepted = verify_canaries(args.result_root, args.platform)
    print(json.dumps({"status": "completed", "platform": args.platform, "results": accepted}, indent=2))


if __name__ == "__main__":
    main()
