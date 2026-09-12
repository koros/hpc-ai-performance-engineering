#!/usr/bin/env python3
"""Verify that both model-specific Transformer Engine FP8 smoke runs passed."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

EXPECTED_RESULTS_BY_PLATFORM = {
    "l40s": {
        "fp8_gpt2_training_acceptance": "gpt2_small",
        "fp8_distilbert_training_acceptance": "distilbert",
    },
    "h100": {
        "fp8_gpt2_training_acceptance_h100": "gpt2_small",
        "fp8_distilbert_training_acceptance_h100": "distilbert",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-gpu-model", required=True)
    parser.add_argument("--expected-git-commit")
    args = parser.parse_args()

    expected_commit = args.expected_git_commit or _git_head()
    platform_id = _platform_id(args.expected_gpu_model)
    expected_results = EXPECTED_RESULTS_BY_PLATFORM.get(platform_id)
    if expected_results is None:
        supported = ", ".join(sorted(EXPECTED_RESULTS_BY_PLATFORM))
        raise SystemExit(
            f"Unsupported expected GPU model {args.expected_gpu_model!r}; FP8 acceptance supports: {supported}"
        )
    issues: list[str] = []
    accepted: dict[str, dict[str, Any]] = {}
    for experiment_id, workload in expected_results.items():
        path = args.result_root / f"{experiment_id}.json"
        if not path.is_file():
            issues.append(f"missing result: {path}")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        metrics = record.get("metrics", {})
        environment = record.get("environment", {})
        prefix = f"{experiment_id}: "
        if record.get("status") != "completed":
            issues.append(prefix + f"status is {record.get('status')!r}: {record.get('error')}")
        if record.get("workload") != workload:
            issues.append(prefix + f"workload is {record.get('workload')!r}")
        if record.get("precision") != "fp8":
            issues.append(prefix + f"precision is {record.get('precision')!r}")
        if metrics.get("precision_backend") != "transformer_engine_delayed_scaling_hybrid":
            issues.append(prefix + "Transformer Engine precision backend was not observed")
        if metrics.get("compute_dtype_observed") != "fp8_mixed_bf16":
            issues.append(prefix + "FP8/BF16 mixed compute was not observed")
        if metrics.get("fp8_forward_enabled") is not True:
            issues.append(prefix + "FP8 forward execution was not enabled")
        if int(metrics.get("transformer_engine_module_count") or 0) < 1:
            issues.append(prefix + "no projection was converted to a Transformer Engine module")
        if not _positive_finite(metrics.get("throughput_tokens_sec")):
            issues.append(prefix + "training throughput is not positive and finite")
        if not _finite(metrics.get("loss")):
            issues.append(prefix + "training loss is not finite")
        if metrics.get("quality_finite") is not True or not _finite(metrics.get("quality_loss")):
            issues.append(prefix + "held-out quality loss is not finite")
        if environment.get("git_dirty") is not False:
            issues.append(prefix + "repository was dirty during acceptance")
        if expected_commit and environment.get("git_commit") != expected_commit:
            issues.append(prefix + f"git commit differs from current acceptance target {expected_commit}")
        if args.expected_gpu_model.lower() not in str(environment.get("observed_gpu_model", "")).lower():
            issues.append(prefix + f"expected GPU token {args.expected_gpu_model!r} was not observed")
        accepted[experiment_id] = {
            "result": str(path),
            "workload": workload,
            "transformer_engine_version": metrics.get("transformer_engine_version"),
            "transformer_engine_module_count": metrics.get("transformer_engine_module_count"),
            "loss": metrics.get("loss"),
            "quality_loss": metrics.get("quality_loss"),
            "throughput_tokens_sec": metrics.get("throughput_tokens_sec"),
        }

    if issues:
        raise SystemExit("FP8 training acceptance failed:\n- " + "\n- ".join(issues))
    print(
        json.dumps(
            {
                "status": "completed",
                "platform": platform_id,
                "git_commit": expected_commit,
                "results": accepted,
            },
            indent=2,
        )
    )


def _platform_id(expected_gpu_model: str) -> str:
    token = expected_gpu_model.lower()
    if "l40s" in token:
        return "l40s"
    if "h100" in token:
        return "h100"
    return ""


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _positive_finite(value: object) -> bool:
    return _finite(value) and float(value) > 0


if __name__ == "__main__":
    main()
