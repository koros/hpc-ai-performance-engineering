"""Study identities for scaling curves, including deliberately shared controls."""

from __future__ import annotations

import json
from typing import Any


def scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("status") != "completed" or row.get("mode") == "profiling":
            continue
        if row.get("parameter_scaling_role") in {"canary", "diagnostic"}:
            continue
        if row.get("scaling_type") not in {"strong", "weak"} and row.get("phase") not in {
            "Strong scaling",
            "Weak scaling",
        }:
            continue
        selected.append(row)
        # One physical trial can support two explicitly declared analyses. It
        # remains one trial in the raw, trial and campaign-count tables.
        weak_batch = row.get("parameter_scaling_weak_per_gpu_batch_size")
        local_batch = row.get("per_gpu_batch_size")
        if weak_batch not in (None, "") and float(weak_batch) == float(local_batch or 0):
            selected.append({**row, "phase": "Weak scaling", "scaling_type": "weak"})
    return selected


def scaling_contract(row: dict[str, Any]) -> str:
    """Keep all fixed execution settings in the grouping key.

    Strong curves fix the effective global batch; weak curves fix the local
    micro-batch and accumulation. GPU count and the derived counterpart of the
    batch are the only shape fields deliberately omitted from each curve.
    """
    weak = row.get("scaling_type") == "weak" or row.get("phase") == "Weak scaling"
    varying = {
        "batch_size",
        "micro_batch_size",
        "per_gpu_batch_size",
        "global_batch_size",
        "effective_batch_size",
        "fixed_workload",
        "scaling_type",
        "scaling_weak_per_gpu_batch_size",
    }
    if not weak and row.get("parameter_scaling_study_id"):
        # A controlled strong curve fixes the global batch and its exact inputs
        # (digest and token count below), so how GPUs and accumulation split that
        # batch is the derived shape, as the local micro-batch already is.
        varying.add("gradient_accumulation_steps")
    fixed = {
        key: str(value)
        for key, value in row.items()
        if key.startswith("parameter_") and key.removeprefix("parameter_") not in varying and value not in (None, "")
    }
    for key in (
        "model_revision",
        "dataset",
        "dataset_revision",
        "measured_iterations",
        "warmup_iterations",
        "metric_dataset_execution",
        "metric_parameter_dtype_observed",
        "metric_scaling_runtime_sha256",
        "environment_torch_version",
        "environment_cuda_version",
        "environment_transformers_version",
        "environment_nccl_version",
        "environment_deepspeed_version",
        "runtime_environment",
    ):
        fixed[key] = str(row.get(key) or "")
    fixed["study_id"] = str(row.get("parameter_scaling_study_id") or "legacy")
    fixed["sequence_length"] = str(row.get("parameter_sequence_length") or row.get("metric_sequence_length") or "")
    batch_key = "per_gpu_batch_size" if weak else "global_batch_size"
    batch = row.get(batch_key) or row.get(f"parameter_{batch_key}")
    fixed[batch_key] = str(float(batch)) if batch not in (None, "") else ""
    if not weak:
        fixed["global_input_sha256"] = str(row.get("metric_global_input_sha256") or "")
        fixed["total_tokens"] = str(row.get("metric_total_tokens") or "")
    return json.dumps(fixed, sort_keys=True, separators=(",", ":"))
