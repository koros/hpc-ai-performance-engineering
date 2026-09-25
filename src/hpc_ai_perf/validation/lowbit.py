"""Fail-closed, platform-specific certification for the low-bit matrix."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from hpc_ai_perf.config.catalog import discover_experiment_configs
from hpc_ai_perf.config.lowbit import ARTIFACTS, METHOD_IDS, digest, runtime_fingerprint
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config


def resolved_lowbit_config(experiment_id: str, platform: str):
    if platform not in {"l40s", "h100"}:
        raise ValueError("Low-bit platform must be l40s or h100")
    entry = next(
        e for e in discover_experiment_configs("configs/lowbit_serving") if e.config.experiment_id == experiment_id
    )
    condition = next(
        c for c in load_platform_conditions("platforms/execution_matrix.json", platform) if c.matrix_id == experiment_id
    )
    return resolve_platform_config(entry, load_platform_profile(f"platforms/{platform}.json"), condition)


def matrix_fingerprint() -> str:
    return digest(
        {e.config.experiment_id: e.config.to_dict() for e in discover_experiment_configs("configs/lowbit_serving")}
    )


def probe_packages(python: str) -> dict[str, Any]:
    # No torch import or GPU allocation is needed to verify installed metadata.
    code = """
import json
from importlib.metadata import version, PackageNotFoundError
packages = {}
for name in ('torch', 'vllm', 'transformers', 'tokenizers', 'datasets', 'huggingface-hub', 'bitsandbytes'):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        packages[name] = None
print(json.dumps(packages))
"""
    environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    result = subprocess.run(
        [python, "-c", code], text=True, capture_output=True, check=True, timeout=30, env=environment
    )
    return json.loads(result.stdout)


def verify_acceptance(root: Path, platform: str, packages: dict[str, Any], *, include_bnb: bool = False) -> None:
    issues = []
    methods = ["bf16", "awq", "gptq"] + (["bitsandbytes_nf4"] if include_bnb else [])
    for method in methods:
        path = root / f"{method}.json"
        if not path.is_file():
            issues.append(f"missing {method} acceptance: {path}")
            continue
        record = json.loads(path.read_text())
        metrics = record.get("metrics", {})
        env = record.get("environment", {})
        config = resolved_lowbit_config(METHOD_IDS[method][1], platform)
        p = config.workload.parameters
        checks = {
            "completed request and quality evaluation": record.get("status") == "completed"
            and record.get("error") is None,
            "platform": platform in str(env.get("observed_gpu_model", "")).lower(),
            "one GPU": record.get("gpus") == 1,
            "runtime fingerprint": metrics.get("study_runtime_sha256") == runtime_fingerprint(),
            "matrix fingerprint": metrics.get("lowbit_matrix_sha256") == matrix_fingerprint(),
            "parameter fingerprint": metrics.get("study_parameters_sha256") == digest(p),
            "artifact revision": metrics.get("hf_model_revision") == config.workload.model_revision,
            "backend": metrics.get("precision_backend") == (ARTIFACTS[method][2] or "vllm_bf16"),
            "quantization method": metrics.get("quantization_method") == method,
            "BF16 compute": metrics.get("compute_dtype_observed") == "torch.bfloat16",
            "runtime dependencies": json.loads(metrics.get("runtime_packages", "{}")) == packages,
            "quality coverage": metrics.get("quality_tokens")
            == p["quality_samples"] * (p["quality_sequence_length"] - 1),
            "KV-cache allocation": metrics.get("kv_cache_blocks_observed") == p["kv_cache_blocks"],
            "per-request timing coverage": metrics.get("latency_sample_count", 0) == metrics.get("requests", -1),
            "energy sampling coverage": energy_coverage_valid(metrics),
        }
        for metric in (
            "throughput_tokens_sec",
            "quality_perplexity",
            "latency_p95_ms",
            "time_to_first_token_p95_ms",
            "memory_model_weights_gb",
        ):
            value = metrics.get(metric)
            checks[metric] = isinstance(value, int | float) and math.isfinite(value) and value > 0
        duration = metrics.get("measured_seconds")
        checks["measurement duration"] = isinstance(duration, int | float) and duration >= p["min_measured_seconds"]
        issues.extend(f"{method}: {label} failed" for label, passed in checks.items() if not passed)
    if issues:
        raise ValueError("Low-bit acceptance failed:\n- " + "\n- ".join(issues))


def energy_coverage_valid(metrics: dict[str, Any]) -> bool:
    """New studies require interior observations, bounded gaps and verified GPUs."""
    values = {
        name: metrics.get(name)
        for name in (
            "energy_joules_measured_region",
            "nvidia_smi.measurement_interior_samples",
            "nvidia_smi.measurement_max_gap_seconds",
        )
    }
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in values.values()):
        return False
    return (
        metrics.get("energy_scope") == "measured_region"
        and metrics.get("energy_measurement_status") == "sufficient"
        and metrics.get("nvidia_smi.gpu_mapping_method") in {"cuda_uuid", "cgroup_visible_rows"}
        and values["energy_joules_measured_region"] > 0
        and values["nvidia_smi.measurement_interior_samples"] >= 10
        and 0 <= values["nvidia_smi.measurement_max_gap_seconds"] <= 1.0
    )
