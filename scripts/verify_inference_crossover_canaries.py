"""Validate the collected same-platform inference crossover canaries."""

import argparse
from pathlib import Path

from hpc_ai_perf.validation.inference_crossover import verify_inference_canaries

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--platform", choices=("h100", "l40s"), required=True)
parser.add_argument("--result-root", type=Path, default=Path("results/raw"))
args = parser.parse_args()
accepted = verify_inference_canaries(args.result_root / args.platform, args.platform)
print(f"Inference crossover acceptance passed: {len(accepted)} {args.platform} canaries")
