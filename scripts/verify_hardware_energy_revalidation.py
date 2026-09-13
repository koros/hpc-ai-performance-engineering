#!/usr/bin/env python3
"""Verify measured-region energy evidence for L40S hardware comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_REFERENCES = {
    "E235": "E123",
    "E236": "E124",
    "E237": "E127",
    "E238": "E128",
}
REQUIRED_TRIALS = 3
VERIFIED_MAPPING_METHODS = {"cuda_uuid", "cgroup_visible_rows"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    issues, accepted = verify_results(args.result_root)
    if issues:
        raise SystemExit("Hardware energy revalidation failed:\n- " + "\n- ".join(issues))
    print(json.dumps({"status": "completed", "results": accepted}, indent=2))


def verify_results(result_root: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    issues: list[str] = []
    accepted: dict[str, dict[str, Any]] = {}
    for experiment_id, reference_id in EXPECTED_REFERENCES.items():
        paths = sorted((result_root / experiment_id).glob("*.json"))
        if len(paths) != REQUIRED_TRIALS:
            issues.append(f"{experiment_id}: expected {REQUIRED_TRIALS} result records, found {len(paths)}")
        valid_repetitions: set[int] = set()
        energies: list[float] = []
        powers: list[float] = []
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            prefix = f"{experiment_id}/{path.name}: "
            metrics = record.get("metrics", {})
            environment = record.get("environment", {})
            metadata = record.get("config", {}).get("metadata", {})
            repetition = record.get("repetition_number")
            record_issues: list[str] = []
            if record.get("status") != "completed":
                record_issues.append(f"status is {record.get('status')!r}: {record.get('error')}")
            if metadata.get("energy_reference_experiment_id") != reference_id:
                record_issues.append(f"does not reference {reference_id}")
            if not isinstance(repetition, int) or repetition not in range(1, REQUIRED_TRIALS + 1):
                record_issues.append(f"invalid repetition number {repetition!r}")
            if metrics.get("energy_scope") != "measured_region":
                record_issues.append(f"energy scope is {metrics.get('energy_scope')!r}")
            if metrics.get("energy_measurement_status") != "sufficient":
                record_issues.append(f"energy measurement status is {metrics.get('energy_measurement_status')!r}")
            energy = metrics.get("energy_joules_measured_region")
            power = metrics.get("power_draw_watts")
            if not _positive_finite(energy):
                record_issues.append("measured-region energy is not positive and finite")
            if not _positive_finite(power):
                record_issues.append("measured-region average power is not positive and finite")
            samples = metrics.get("nvidia_smi.measurement_samples")
            if not isinstance(samples, int) or samples < 2:
                record_issues.append(f"measured region has insufficient samples: {samples!r}")
            mapping = metrics.get("nvidia_smi.gpu_mapping_method")
            if mapping not in VERIFIED_MAPPING_METHODS:
                record_issues.append(f"GPU telemetry mapping is not verified: {mapping!r}")
            if str(metrics.get("nvidia_smi.telemetry_schema_version")) != "3":
                record_issues.append("GPU telemetry schema is not version 3")
            if "l40s" not in str(environment.get("observed_gpu_model", "")).lower():
                record_issues.append("the observed GPU is not an L40S")
            if record_issues:
                issues.extend(prefix + issue for issue in record_issues)
                continue
            valid_repetitions.add(repetition)
            energies.append(float(energy))
            powers.append(float(power))
        if valid_repetitions != {1, 2, 3}:
            issues.append(
                f"{experiment_id}: valid measured-region repetitions are "
                f"{sorted(valid_repetitions)}, expected [1, 2, 3]"
            )
        if energies:
            accepted[experiment_id] = {
                "energy_reference_experiment_id": reference_id,
                "trials": len(energies),
                "avg_energy_joules": sum(energies) / len(energies),
                "avg_power_watts": sum(powers) / len(powers),
            }
    return issues, accepted


def _positive_finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0


if __name__ == "__main__":
    main()
