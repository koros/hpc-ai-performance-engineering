"""Require collected current-runtime serving canaries before crossover trials."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.inference_crossover import CANARY_WAVE, serving_runtime_sha256
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.serving_layout import declared_serving_layout
from hpc_ai_perf.validation import validate_result_file
from hpc_ai_perf.validation.qwen7b_training import canary_contract_sha256
from hpc_ai_perf.validation.scaling_acceptance import _manifest_entries


def verify_inference_canaries(result_root: str | Path, platform: str) -> dict[str, str]:
    root = Path(result_root)
    checksums = _manifest_entries(root / "SHA256SUMS")
    entries = {e.matrix_id: e.config for e in discover_experiment_configs("configs/inference_crossover")}
    conditions = load_platform_conditions("platforms/execution_matrix.json", platform, wave=CANARY_WAVE)
    if not conditions:
        raise ValueError(f"No inference crossover canary policy for {platform}")
    accepted = {}
    for condition in conditions:
        config = entries[condition.matrix_id]
        for path in sorted((root / condition.matrix_id).glob("*.json")):
            record = json.loads(path.read_text())
            if checksums.get(path.relative_to(root).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
                continue
            if validate_result_file(path) or not isinstance(record.get("config"), dict):
                continue
            if canary_contract_sha256(record["config"]) != canary_contract_sha256(config):
                continue
            if not canary_issues(record, config, platform):
                accepted[condition.matrix_id] = str(path)
                break
    missing = sorted({c.matrix_id for c in conditions} - accepted.keys())
    if missing:
        raise ValueError("Current-runtime inference canaries missing or invalid: " + ", ".join(missing))
    return accepted


def canary_issues(record, config, platform) -> list[str]:
    m = record.get("metrics") or {}
    p = config.workload.parameters
    env = record.get("environment") or {}
    expected_requests = config.runtime.max_steps * p["concurrent_requests"]
    checks = {
        "completion": record.get("status") == "completed" and record.get("error") is None,
        "platform": record.get("platform_id") == platform,
        "clean source": record.get("git_dirty") is False and env.get("git_dirty") is False,
        "GPU count": record.get("gpus") == config.hardware.gpus,
        "runtime": m.get("serving_runtime_sha256") == serving_runtime_sha256(),
        "layout": m.get("serving_layout_observed") == declared_serving_layout(p),
        "TP degree": m.get("tensor_parallel_size_observed") == p["tensor_parallel_size"],
        "replicas": m.get("replica_count_observed") == p["replica_count"],
        "requests": m.get("requests") == expected_requests,
        "fixed output work": m.get("generated_tokens") == expected_requests * p["max_new_tokens"],
        "throughput": _positive(m.get("throughput_tokens_sec")),
        "TTFT timestamps": _positive(m.get("time_to_first_token_p95_ms")),
        "decode timestamps": _positive(m.get("time_per_output_token_p95_ms")),
        "memory telemetry": _positive(m.get("memory_used_gb", record.get("memory_used_gb"))),
    }
    if p["replica_count"] > 1:
        checks["routing count"] = m.get("replica_request_count_total") == expected_requests
        checks["routing balance"] = m.get("replica_request_count_imbalance_pct") == 0
    return [name for name, passed in checks.items() if not passed]


def _positive(value):
    import math

    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
