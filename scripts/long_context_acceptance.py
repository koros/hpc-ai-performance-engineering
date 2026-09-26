"""Certify the 8k Qwen2.5-7B context shape before canonical trials."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from hpc_ai_perf.config.long_context import CONTEXT_IDS
from hpc_ai_perf.runners.local import LocalBenchmarkRunner
from hpc_ai_perf.validation.long_context import (
    matrix_fingerprint,
    resolved_long_context_config,
    verify_acceptance,
)
from hpc_ai_perf.validation.lowbit import probe_packages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "verify"))
    parser.add_argument("--platform", choices=("l40s", "h100"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "worker":
        config = resolved_long_context_config(CONTEXT_IDS[8192], args.platform)
        config = replace(
            config,
            metadata={**config.metadata, "evidence_role": "diagnostic_only"},
        )
        result = LocalBenchmarkRunner().run(config)
        result.metrics["long_context_matrix_sha256"] = matrix_fingerprint()
        with (args.result_root / "long_context_8192.json").open("x") as handle:
            json.dump(result.to_dict(), handle, indent=2)
            handle.write("\n")
        if result.status != "completed":
            raise SystemExit(result.error)
        return
    if args.action == "run":
        args.result_root.mkdir(parents=True, exist_ok=False)
        subprocess.run(
            [
                sys.executable,
                __file__,
                "worker",
                "--platform",
                args.platform,
                "--result-root",
                str(args.result_root),
            ],
            check=True,
        )
    verify_acceptance(args.result_root, args.platform, probe_packages(sys.executable))
    print(f"Long-context acceptance passed for {args.platform}: {args.result_root}")


if __name__ == "__main__":
    main()
