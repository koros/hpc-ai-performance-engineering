"""Validate 16/64/128 MiB transfer reports and save a checksum-linked evidence manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc_ai_perf.validation.nvlink_calibration import verify_calibrations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("build/nvlink_calibration/h100"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.name == "CALIBRATION.json":
        parser.error("Output must not overwrite a source CALIBRATION.json report")
    # Remove an obsolete manifest before checking new reports, so failure cannot leave stale acceptance.
    args.output.unlink(missing_ok=True)
    manifest = verify_calibrations(args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("NVLink KiB-to-byte conversion passed: 16, 64 and 128 MiB; both directions")
    print(f"Manifest: {args.output}")
    print("Inference collector timing and profiling overhead are not yet validated.")


if __name__ == "__main__":
    main()
