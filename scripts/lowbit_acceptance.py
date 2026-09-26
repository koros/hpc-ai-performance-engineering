"""Run diagnostic gates in fresh processes; never import them as canonical trial evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from hpc_ai_perf.config.lowbit import METHOD_IDS, QUALITY_METHOD_IDS
from hpc_ai_perf.runners.local import LocalBenchmarkRunner
from hpc_ai_perf.validation.lowbit import (
    matrix_fingerprint,
    probe_packages,
    resolved_lowbit_config,
    verify_acceptance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "verify"))
    parser.add_argument("--platform", choices=("l40s", "h100"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--include-bnb", action="store_true")
    parser.add_argument("--wave", choices=("lowbit-serving", "lowbit-bnb", "lowbit-quality"), default="lowbit-serving")
    parser.add_argument("--method", choices=METHOD_IDS)
    args = parser.parse_args()
    if args.action == "worker":
        if args.method is None:
            parser.error("worker requires --method")
        eid = QUALITY_METHOD_IDS[args.method] if args.wave == "lowbit-quality" else METHOD_IDS[args.method][1]
        config = resolved_lowbit_config(eid, args.platform)
        config = replace(
            config,
            experiment_id=f"acceptance_lowbit_{args.method}_{args.platform}",
            metadata={**config.metadata, "evidence_role": "diagnostic_only"},
        )
        result = LocalBenchmarkRunner().run(config)
        result.metrics["lowbit_matrix_sha256"] = matrix_fingerprint(args.wave)
        with (args.result_root / f"{args.method}.json").open("x") as handle:
            json.dump(result.to_dict(), handle, indent=2)
            handle.write("\n")
        if result.status != "completed":
            raise SystemExit(result.error)
        return
    if args.action == "run":
        # Each Slurm job gets a fresh directory; never accept stale siblings from
        # a previous failed run. Each method gets a fresh CUDA worker process.
        args.result_root.mkdir(parents=True, exist_ok=False)
        methods = ["bf16", "awq", "gptq"] + (
            ["bitsandbytes_nf4"] if args.include_bnb or args.wave == "lowbit-quality" else []
        )
        for method in methods:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "worker",
                    "--platform",
                    args.platform,
                    "--result-root",
                    str(args.result_root),
                    "--method",
                    method,
                    "--wave",
                    args.wave,
                ],
                check=True,
            )
    verify_acceptance(
        args.result_root, args.platform, probe_packages(sys.executable), include_bnb=args.include_bnb, wave=args.wave
    )
    print(f"Low-bit acceptance passed for {args.platform}: {args.result_root}")


if __name__ == "__main__":
    main()
