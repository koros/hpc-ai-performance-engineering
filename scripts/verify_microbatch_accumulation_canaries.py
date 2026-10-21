"""Verify collected L40S microbatch/accumulation acceptance results."""

import argparse
from pathlib import Path

from hpc_ai_perf.validation.microbatch_accumulation import verify_microbatch_canaries

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result-root", type=Path, default=Path("results/raw/l40s"))
args = parser.parse_args()
accepted = verify_microbatch_canaries(args.result_root)
print(f"Microbatch accumulation acceptance passed: {len(accepted)} L40S canaries")
