"""Check collected H100 energy anchors without altering their immutable evidence."""

import argparse
import json
import math
from pathlib import Path

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.validation.lowbit import energy_coverage_valid
from hpc_ai_perf.validation.qwen7b_training import canary_contract_sha256, matching_records


def verify_results(root: Path) -> dict:
    entries = {e.config.experiment_id: e for e in discover_experiment_configs("configs/hardware_energy_revalidation")}
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json", "h100", wave="h100-inference-energy-anchors"
    )
    issues, accepted = [], {}
    for condition in conditions:
        config = resolve_platform_config(
            entries[condition.matrix_id], load_platform_profile("platforms/h100.json"), condition
        )
        matches = matching_records(root / condition.matrix_id, canary_contract_sha256(config))
        valid = {}
        for path, record in matches:
            metrics = record.get("metrics", {})
            if (
                record.get("status") == "completed"
                and record.get("error") is None
                and isinstance(metrics.get("measured_seconds"), int | float)
                and math.isfinite(metrics["measured_seconds"])
                and metrics["measured_seconds"] >= 30
                and energy_coverage_valid(metrics)
                and "h100" in str(record.get("environment", {}).get("observed_gpu_model", "")).lower()
            ):
                valid[record.get("repetition_number")] = str(path)
        if set(valid) != {1, 2, 3}:
            issues.append(
                f"{condition.matrix_id}: requires three current-contract trials with duration and energy coverage"
            )
        accepted[condition.matrix_id] = valid
    if issues:
        raise ValueError("H100 energy anchor verification failed:\n- " + "\n- ".join(issues))
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/raw/h100"))
    args = parser.parse_args()
    print(json.dumps({"status": "completed", "results": verify_results(args.result_root)}, indent=2))


if __name__ == "__main__":
    main()
