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
    assert [r[0] for r in rows] == [f"E{i}" for i in range(750, 769)]
    assert len([r for r in rows if r[2] == CANARY_WAVE]) == 2
    assert len([r for r in rows if r[2] == PERFORMANCE_WAVE]) * 3 == 12
    entries = discover_experiment_configs(CONFIG_ROOT)
    for entry in entries:
        assert capability_issues(entry.config) == []
        assert entry.config.hardware.gpus == 8
        assert entry.config.workload.model == "Qwen/Qwen2.5-7B-Instruct"
        p = entry.config.workload.parameters
        assert p["tensor_parallel_size"] * p["replica_count"] == 8
        assert p["concurrent_requests"] in (1, 4, 8) and p["max_new_tokens"] == 128
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


def test_extension_preserves_existing_controls_and_valid_head_partitioning():
    rows = design()
    extension = rows[6:9]
    assert [r[0] for r in extension] == ["E756", "E757", "E758"]
    assert [r[3]["workload"]["parameters"]["prompt_length"] for r in extension] == [1280, 128, 1280]
    for _, _, _, data in rows:
        p = data["workload"]["parameters"]
        assert 28 % p["tensor_parallel_size"] == 0  # pinned model query heads
        assert p["tensor_parallel_size"] in (1, 2, 4)
    for row, control in zip(extension[1:], rows[2:5:2], strict=True):
        p = row[3]["workload"]["parameters"]
        assert p["comparison_control_id"] == control[0]
        assert p["comparison_scope"] == control[3]["workload"]["parameters"]["comparison_scope"]
        assert row[3]["workload"]["model_revision"] == control[3]["workload"]["model_revision"]


def test_extension_performance_requires_original_and_new_acceptance(monkeypatch):
    from hpc_ai_perf.config.qwen_layout import EXTENSION_CANARY_WAVE, EXTENSION_PERFORMANCE_WAVE
    from hpc_ai_perf.validation import inference_crossover as gate

    waves = []

    def check(*args, **kwargs):
        waves.append(kwargs["canary_wave"])
        if kwargs["canary_wave"] == EXTENSION_CANARY_WAVE:
            raise ValueError("TP4 acceptance absent")

    monkeypatch.setattr(gate, "verify_inference_canaries", check)
    with pytest.raises(ValueError, match="TP4 acceptance absent"):
        _submit_platform_wave(
            SimpleNamespace(
                wave=EXTENSION_PERFORMANCE_WAVE,
                platform_profile="platforms/l40s.json",
                venv_activate="/private/activate",
            )
        )
    assert waves == [CANARY_WAVE, EXTENSION_CANARY_WAVE]


def test_singleton_acceptance_exercises_every_replica_and_preserves_controls():
    from collections import Counter

    from hpc_ai_perf.config.qwen_layout import LOW_CONCURRENCY_CANARY_WAVE, LOW_CONCURRENCY_PERFORMANCE_WAVE
    from hpc_ai_perf.workloads.vllm_router import route_prompt_indices

    rows = design()
    low = rows[9:15]
    assert [row[0] for row in low] == [f"E{i}" for i in range(759, 765)]
    for eid, _, wave, data in low:
        p = data["workload"]["parameters"]
        assert p["concurrent_requests"] == 1 and p["prompt_length"] == 1280
        assert p["max_new_tokens"] == 128
        assert (
            p["comparison_scope"] == rows[4][3]["workload"]["parameters"]["comparison_scope"]
            if wave == LOW_CONCURRENCY_PERFORMANCE_WAVE
            else p["comparison_scope"].endswith("_canary")
        )
        if wave == LOW_CONCURRENCY_CANARY_WAVE:
            assert data["runtime"]["max_steps"] == 8
            counts = Counter()
            for round_number in range(8):
                groups = route_prompt_indices(1, p["replica_count"], starting_replica=round_number % p["replica_count"])
                counts.update({i: len(group) for i, group in enumerate(groups)})
            assert len(set(counts.values())) == 1 and sum(counts.values()) == 8
        else:
            assert data["runtime"]["max_steps"] == 30
            assert data["metadata"]["concurrency_reference_id"] in ("E754", "E755", "E758")
            assert p.get("comparison_control_id") == (None if eid == "E762" else "E762")


def test_singleton_performance_requires_shape_specific_acceptance(monkeypatch):
    from hpc_ai_perf.config.qwen_layout import LOW_CONCURRENCY_CANARY_WAVE, LOW_CONCURRENCY_PERFORMANCE_WAVE
    from hpc_ai_perf.validation import inference_crossover as gate

    waves = []

    def check(*args, **kwargs):
        waves.append(kwargs["canary_wave"])
        if kwargs["canary_wave"] == LOW_CONCURRENCY_CANARY_WAVE:
            raise ValueError("Concurrency-one acceptance absent")

    monkeypatch.setattr(gate, "verify_inference_canaries", check)
    with pytest.raises(ValueError, match="Concurrency-one acceptance absent"):
        _submit_platform_wave(
            SimpleNamespace(
                wave=LOW_CONCURRENCY_PERFORMANCE_WAVE,
                platform_profile="platforms/l40s.json",
                venv_activate="/private/activate",
            )
        )
    assert len(waves) == 3 and waves[-1] == LOW_CONCURRENCY_CANARY_WAVE


def test_concurrency_four_preserves_shape_and_compares_tp2_with_tp4():
    from hpc_ai_perf.config.qwen_layout import MID_CONCURRENCY_CANARY_WAVE

    rows = design()
    mid = rows[15:]
    assert [r[0] for r in mid] == ["E765", "E766", "E767", "E768"]
    for eid, _, wave, data in mid:
        p = data["workload"]["parameters"]
        assert p["concurrent_requests"] == 4 and p["prompt_length"] == 1280
        assert p["max_new_tokens"] == 128 and p["tensor_parallel_size"] in (2, 4)
        assert data["hardware"]["gpus"] == 8
        assert data["runtime"]["max_steps"] == (2 if wave == MID_CONCURRENCY_CANARY_WAVE else 30)
        requests = data["runtime"]["max_steps"] * 4
        assert requests % p["replica_count"] == 0
        assert p.get("comparison_control_id") == ("E767" if eid == "E768" else None)
        assert len(data["metadata"]["concurrency_reference_ids"]) == 2


def test_concurrency_four_performance_requires_new_acceptance(monkeypatch):
    from hpc_ai_perf.config.qwen_layout import MID_CONCURRENCY_CANARY_WAVE, MID_CONCURRENCY_PERFORMANCE_WAVE
    from hpc_ai_perf.validation import inference_crossover as gate

    waves = []

    def check(*args, **kwargs):
        waves.append(kwargs["canary_wave"])
        if kwargs["canary_wave"] == MID_CONCURRENCY_CANARY_WAVE:
            raise ValueError("Concurrency-four acceptance absent")

    monkeypatch.setattr(gate, "verify_inference_canaries", check)
    with pytest.raises(ValueError, match="Concurrency-four acceptance absent"):
        _submit_platform_wave(
            SimpleNamespace(
                wave=MID_CONCURRENCY_PERFORMANCE_WAVE,
                platform_profile="platforms/l40s.json",
                venv_activate="/private/activate",
            )
        )
    assert len(waves) == 4 and waves[-1] == MID_CONCURRENCY_CANARY_WAVE
