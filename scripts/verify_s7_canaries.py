"""Verify the eight collected current-runtime S7 real-text canaries."""

import argparse
from pathlib import Path

from hpc_ai_perf.validation.scaling_deep_dive import verify_deep_dive_canaries

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result-root", type=Path, default=Path("results/raw/l40s"))
args = parser.parse_args()
accepted = verify_deep_dive_canaries(args.result_root, "l40s", canary_wave="s7-real-text-canary")
print(f"S7 acceptance passed: {len(accepted)} L40S real-text canaries")
