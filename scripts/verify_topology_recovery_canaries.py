"""Verify collected L40S topology recovery acceptance results."""

import argparse
from pathlib import Path

from hpc_ai_perf.validation.topology_recovery import verify_topology_canaries

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--result-root", type=Path, default=Path("results/raw/l40s"))
args = parser.parse_args()
accepted = verify_topology_canaries(args.result_root)
print(f"Topology recovery acceptance passed: {len(accepted)} L40S canaries")
