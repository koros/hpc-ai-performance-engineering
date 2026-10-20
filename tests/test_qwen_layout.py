import hashlib
import json
from types import SimpleNamespace

import pytest

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.inference_crossover import serving_runtime_sha256
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.config.qwen_layout import CANARY_WAVE, CONFIG_ROOT, PERFORMANCE_WAVE, design
from hpc_ai_perf.validation import capability_issues
from hpc_ai_perf.validation.inference_crossover import canary_issues, verify_inference_canaries


def test_design_is_focused_and_supported():
    rows = design()
    assert [r[0] for r in rows] == [f"E{i}" for i in range(750, 756)]
    assert len([r for r in rows if r[2] == CANARY_WAVE]) == 2
    assert len([r for r in rows if r[2] == PERFORMANCE_WAVE]) * 3 == 12
    entries = discover_experiment_configs(CONFIG_ROOT)
    for entry in entries:
        assert capability_issues(entry.config) == []
        assert entry.config.hardware.gpus == 8
        assert entry.config.workload.model == "Qwen/Qwen2.5-7B-Instruct"
        p = entry.config.workload.parameters
        assert p["tensor_parallel_size"] * p["replica_count"] == 8
        assert p["concurrent_requests"] == 8 and p["max_new_tokens"] == 128
        assert p["require_request_timestamps"] and p["ignore_eos"]
        assert entry.config.runtime.environment["VLLM_USE_V1"] == "0"


def test_performance_gate_runs_before_bundle_creation(monkeypatch):
    from hpc_ai_perf.validation import inference_crossover as gate

    def reject(*args, **kwargs):
        assert kwargs == {"config_root": CONFIG_ROOT, "canary_wave": CANARY_WAVE}
        raise ValueError("Qwen acceptance absent")

    monkeypatch.setattr(gate, "verify_inference_canaries", reject)
    with pytest.raises(ValueError, match="Qwen acceptance absent"):
        _submit_platform_wave(
            SimpleNamespace(
                wave=PERFORMANCE_WAVE, platform_profile="platforms/l40s.json", venv_activate="/private/activate"
            )
        )


def test_qwen_requires_model_specific_collected_evidence(tmp_path, monkeypatch):
    from hpc_ai_perf.validation import inference_crossover as gate

    monkeypatch.setattr(gate, "validate_result_file", lambda path: [])
    entries = {e.matrix_id: e for e in discover_experiment_configs(CONFIG_ROOT)}
    profile = load_platform_profile("platforms/l40s.json")
    manifest = []
    for condition in load_platform_conditions("platforms/execution_matrix.json", "l40s", wave=CANARY_WAVE):
        c = resolve_platform_config(entries[condition.matrix_id], profile, condition)
        p = c.workload.parameters
        record = {
            "status": "completed",
            "error": None,
            "config": c.to_dict(),
            "platform_id": "l40s",
            "gpus": 8,
            "git_dirty": False,
            "environment": {"git_dirty": False},
            "metrics": {
                "serving_runtime_sha256": serving_runtime_sha256(),
                "serving_layout_observed": p["serving_layout"],
                "tensor_parallel_size_observed": p["tensor_parallel_size"],
                "replica_count_observed": p["replica_count"],
                "requests": 16,
                "generated_tokens": 2048,
                "throughput_tokens_sec": 100,
                "time_to_first_token_p95_ms": 10,
                "time_per_output_token_p95_ms": 5,
                "memory_used_gb": 40,
                "replica_request_count_total": 16,
                "replica_request_count_imbalance_pct": 0,
            },
        }
        assert canary_issues(record, c, "l40s") == []
        path = tmp_path / condition.matrix_id / "run.json"
        path.parent.mkdir()
        path.write_text(json.dumps(record))
        manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path).as_posix()}\n")
    (tmp_path / "SHA256SUMS").write_text("".join(manifest))
    assert len(verify_inference_canaries(tmp_path, "l40s", config_root=CONFIG_ROOT, canary_wave=CANARY_WAVE)) == 2
    # A checksum-valid result from another model must never unlock Qwen performance.
    path = next((tmp_path / "E750").glob("*.json"))
    record = json.loads(path.read_text())
    record["config"]["workload"]["model"] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    path.write_text(json.dumps(record))
    manifest[0] = f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path).as_posix()}\n"
    (tmp_path / "SHA256SUMS").write_text("".join(manifest))
    with pytest.raises(ValueError, match="resolved configuration contract"):
        verify_inference_canaries(tmp_path, "l40s", config_root=CONFIG_ROOT, canary_wave=CANARY_WAVE)
