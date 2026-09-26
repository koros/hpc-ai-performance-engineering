"""Fail-closed acceptance for the maximum long-context serving shape."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from hpc_ai_perf.config.catalog import discover_experiment_configs
from hpc_ai_perf.config.long_context import CONTEXT_IDS, runtime_fingerprint
from hpc_ai_perf.config.lowbit import BASE_REVISION, digest
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.validation.lowbit import energy_coverage_valid


def resolved_long_context_config(experiment_id: str, platform: str):
    if platform not in {"l40s", "h100"}:
        raise ValueError("Long-context platform must be l40s or h100")
    entry = next(
        e
        for e in discover_experiment_configs("configs/long_context_serving")
        if e.config.experiment_id == experiment_id
    )
    condition = next(
        c for c in load_platform_conditions("platforms/execution_matrix.json", platform) if c.matrix_id == experiment_id
    )
    return resolve_platform_config(entry, load_platform_profile(f"platforms/{platform}.json"), condition)


def matrix_fingerprint() -> str:
    return digest(
        {
            e.config.experiment_id: e.config.to_dict()
            for e in discover_experiment_configs("configs/long_context_serving")
        }
    )


def verify_acceptance(root: Path, platform: str, packages: dict[str, Any]) -> None:
    path = root / "long_context_8192.json"
    if not path.is_file():
        raise ValueError(f"Long-context acceptance failed:\n- missing acceptance result: {path}")
    record = json.loads(path.read_text())
    metrics = record.get("metrics", {})
    environment = record.get("environment", {})
    config = resolved_long_context_config(CONTEXT_IDS[8192], platform)
    p = config.workload.parameters
    checks = {
        "completed request measurement": record.get("status") == "completed" and record.get("error") is None,
        "platform": platform in str(environment.get("observed_gpu_model", "")).lower(),
        "one GPU": record.get("gpus") == 1,
        "runtime fingerprint": metrics.get("long_context_runtime_sha256") == runtime_fingerprint(),
        "matrix fingerprint": metrics.get("long_context_matrix_sha256") == matrix_fingerprint(),
        "parameter fingerprint": metrics.get("long_context_parameters_sha256") == digest(p),
        "artifact revision": metrics.get("hf_model_revision") == BASE_REVISION,
        "unquantized BF16": metrics.get("vllm_quantization") is None
        and metrics.get("compute_dtype_observed") == "torch.bfloat16",
        "runtime dependencies": json.loads(metrics.get("runtime_packages", "{}")) == packages,
        "8k input": metrics.get("prompt_tokens_per_request") == 8192,
        "fixed context limit": metrics.get("max_model_len_observed") == p["max_model_len"],
        "fixed KV-cache": metrics.get("kv_cache_blocks_observed") == p["kv_cache_blocks"],
        "per-request timing coverage": metrics.get("latency_sample_count", 0) == metrics.get("requests", -1),
        "energy sampling coverage": energy_coverage_valid(metrics),
    }
    for metric in (
        "throughput_tokens_sec",
        "latency_p95_ms",
        "time_to_first_token_p95_ms",
        "memory_model_weights_gb",
    ):
        value = metrics.get(metric)
        checks[metric] = isinstance(value, int | float) and math.isfinite(value) and value > 0
    duration = metrics.get("measured_seconds")
    checks["measurement duration"] = isinstance(duration, int | float) and duration >= p["min_measured_seconds"]
    issues = [label for label, passed in checks.items() if not passed]
    if issues:
        raise ValueError("Long-context acceptance failed:\n- " + "\n- ".join(issues))
