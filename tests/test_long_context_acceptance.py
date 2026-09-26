from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config.long_context import CONTEXT_IDS, runtime_fingerprint
from hpc_ai_perf.config.lowbit import digest
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.validation.long_context import matrix_fingerprint, resolved_long_context_config, verify_acceptance


def write_acceptance(root, platform="l40s", **metric_overrides):
    config = resolved_long_context_config(CONTEXT_IDS[8192], platform)
    record = {
        "status": "completed",
        "error": None,
        "gpus": 1,
        "environment": {"observed_gpu_model": f"NVIDIA {platform.upper()}"},
        "metrics": {
            "long_context_runtime_sha256": runtime_fingerprint(),
            "long_context_matrix_sha256": matrix_fingerprint(),
            "long_context_parameters_sha256": digest(config.workload.parameters),
            "hf_model_revision": config.workload.model_revision,
            "vllm_quantization": None,
            "compute_dtype_observed": "torch.bfloat16",
            "runtime_packages": '{"vllm":"0.10.2"}',
            "prompt_tokens_per_request": 8192,
            "max_model_len_observed": 9216,
            "kv_cache_blocks_observed": 1024,
            "requests": 4,
            "latency_sample_count": 4,
            "throughput_tokens_sec": 20,
            "latency_p95_ms": 200,
            "time_to_first_token_p95_ms": 50,
            "memory_model_weights_gb": 15,
            "measured_seconds": 30,
            "energy_scope": "measured_region",
            "energy_measurement_status": "sufficient",
            "energy_joules_measured_region": 1000,
            "nvidia_smi.gpu_mapping_method": "cuda_uuid",
            "nvidia_smi.measurement_interior_samples": 100,
            "nvidia_smi.measurement_max_gap_seconds": 0.3,
            **metric_overrides,
        },
    }
    (root / "long_context_8192.json").write_text(json.dumps(record))


@pytest.mark.parametrize("platform", ["l40s", "h100"])
def test_long_context_wave_is_platform_separated_and_exact(platform):
    conditions = load_platform_conditions("platforms/execution_matrix.json", platform, wave="vllm-long-context")
    assert {condition.matrix_id for condition in conditions} == {"E247", "E248", "E249"}
    assert {condition.required_trials for condition in conditions} == {3}
    assert {condition.comparability for condition in conditions} == {"exact"}
    for condition in conditions:
        config = resolved_long_context_config(condition.matrix_id, platform)
        assert config.hardware.gpus == 1
        assert capability_issues(config) == []


@pytest.mark.parametrize(
    "metric,value",
    [
        ("prompt_tokens_per_request", 4096),
        ("long_context_matrix_sha256", "old"),
        ("latency_sample_count", 0),
        ("measured_seconds", 0.1),
        ("nvidia_smi.measurement_interior_samples", 0),
    ],
)
def test_long_context_gate_rejects_stale_or_partial_8k_evidence(tmp_path, metric, value):
    write_acceptance(tmp_path, **{metric: value})
    with pytest.raises(ValueError, match="acceptance failed"):
        verify_acceptance(tmp_path, "l40s", {"vllm": "0.10.2"})


def test_long_context_gate_requires_matching_platform_runtime_and_private_wrapper(tmp_path, monkeypatch):
    write_acceptance(tmp_path)
    verify_acceptance(tmp_path, "l40s", {"vllm": "0.10.2"})
    with pytest.raises(ValueError, match="platform"):
        verify_acceptance(tmp_path, "h100", {"vllm": "0.10.2"})
    with pytest.raises(ValueError, match="runtime dependencies"):
        verify_acceptance(tmp_path, "l40s", {"vllm": "0.11.0"})
    monkeypatch.delenv("HPC_AI_LONG_CONTEXT_ACCEPTANCE_DIR", raising=False)
    args = SimpleNamespace(wave="vllm-long-context", platform_profile="platforms/l40s.json", venv_activate=None)
    with pytest.raises(ValueError, match="platform-specific acceptance"):
        _submit_platform_wave(args)
