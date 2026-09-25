from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config.lowbit import ARTIFACTS, METHOD_IDS, digest, runtime_fingerprint
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.validation.lowbit import matrix_fingerprint, resolved_lowbit_config, verify_acceptance


def write_acceptance(root, platform="l40s", **metric_overrides):
    for method in ("bf16", "awq", "gptq", "bitsandbytes_nf4"):
        cfg = resolved_lowbit_config(METHOD_IDS[method][1], platform)
        record = {
            "status": "completed",
            "error": None,
            "gpus": 1,
            "environment": {"observed_gpu_model": f"NVIDIA {platform.upper()}"},
            "metrics": {
                "study_runtime_sha256": runtime_fingerprint(),
                "lowbit_matrix_sha256": matrix_fingerprint(),
                "study_parameters_sha256": digest(cfg.workload.parameters),
                "hf_model_revision": cfg.workload.model_revision,
                "precision_backend": ARTIFACTS[method][2] or "vllm_bf16",
                "quantization_method": method,
                "compute_dtype_observed": "torch.bfloat16",
                "runtime_packages": '{"vllm":"0.10.2"}',
                "quality_tokens": 32 * 255,
                "kv_cache_blocks_observed": 1024,
                "requests": 16,
                "latency_sample_count": 16,
                "throughput_tokens_sec": 20,
                "quality_perplexity": 10,
                "latency_p95_ms": 200,
                "time_to_first_token_p95_ms": 50,
                "memory_model_weights_gb": 5,
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
        (root / f"{method}.json").write_text(json.dumps(record))


@pytest.mark.parametrize("platform", ["h100", "l40s"])
def test_new_conditions_are_separate_single_gpu_waves_with_clean_controls(platform):
    core = load_platform_conditions("platforms/execution_matrix.json", platform, wave="lowbit-serving")
    bnb = load_platform_conditions("platforms/execution_matrix.json", platform, wave="lowbit-bnb")
    assert {c.matrix_id for c in core} == {f"E{i}" for i in range(239, 245)}
    assert {c.matrix_id for c in bnb} == {"E245", "E246"}
    for condition in core + bnb:
        config = resolved_lowbit_config(condition.matrix_id, platform)
        assert config.hardware.gpus == 1
        assert config.hardware.platform == ("lovelace_h100" if platform == "h100" else "lovelace_l40s")
        assert condition.required_trials == 3
        assert capability_issues(config) == []
        control = resolved_lowbit_config(config.workload.parameters["comparison_control_id"], platform)
        for key, value in config.workload.parameters.items():
            if key != "quantization_method":
                assert control.workload.parameters[key] == value


def test_gate_checks_all_methods_and_rejects_stale_platform_or_dependency_evidence(tmp_path):
    write_acceptance(tmp_path)
    verify_acceptance(tmp_path, "l40s", {"vllm": "0.10.2"}, include_bnb=True)
    with pytest.raises(ValueError, match="platform failed"):
        verify_acceptance(tmp_path, "h100", {"vllm": "0.10.2"})
    with pytest.raises(ValueError, match="runtime dependencies"):
        verify_acceptance(tmp_path, "l40s", {"vllm": "0.11.0"})
    write_acceptance(tmp_path, study_runtime_sha256="old")
    with pytest.raises(ValueError, match="runtime fingerprint"):
        verify_acceptance(tmp_path, "l40s", {"vllm": "0.10.2"})


@pytest.mark.parametrize(
    "metric,value",
    [
        ("quality_perplexity", float("nan")),
        ("quality_tokens", 1),
        ("measured_seconds", 0.1),
        ("latency_sample_count", 0),
        ("lowbit_matrix_sha256", "old"),
        ("nvidia_smi.measurement_interior_samples", 0),
        ("nvidia_smi.measurement_max_gap_seconds", 12),
    ],
)
def test_gate_does_not_accept_partial_or_invalid_measurements(tmp_path, metric, value):
    write_acceptance(tmp_path, **{metric: value})
    with pytest.raises(ValueError, match="acceptance failed"):
        verify_acceptance(tmp_path, "l40s", {"vllm": "0.10.2"})


def test_missing_gate_prevents_generic_submit_before_any_sbatch(monkeypatch):
    monkeypatch.delenv("HPC_AI_LOWBIT_ACCEPTANCE_DIR", raising=False)
    args = SimpleNamespace(wave="lowbit-serving", platform_profile="platforms/l40s.json", venv_activate=None)
    with pytest.raises(ValueError, match="platform-specific acceptance"):
        _submit_platform_wave(args)
