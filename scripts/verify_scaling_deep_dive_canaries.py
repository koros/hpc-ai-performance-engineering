"""Verify collected scaling deep-dive canaries before performance submissions."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpc_ai_perf.validation.scaling_deep_dive import verify_deep_dive_canaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("h100", "l40s"), required=True)
    parser.add_argument("--result-root", type=Path, default=Path("results/raw"))
    args = parser.parse_args()
    accepted = verify_deep_dive_canaries(args.result_root / args.platform, args.platform)
    print(f"Scaling deep-dive acceptance passed for {args.platform}: {len(accepted)} collected canaries")


if __name__ == "__main__":
    main()
