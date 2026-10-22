import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.generate_s7_acceptance import expected_matrix

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.config.s7_real_text import CANARY_WAVE, CONFIG_ROOT, design
from hpc_ai_perf.config.scaling import scaling_issues
from hpc_ai_perf.validation import scaling_deep_dive as gate


def test_focused_design_covers_both_models_and_all_batching_modes():
    rows = design()
    assert [eid for eid, _ in rows] == [f"E{i}" for i in range(810, 818)]
    pairs = {(d["workload"]["name"], d["workload"]["parameters"]["text_batching"]) for _, d in rows}
    assert pairs == {
        (w, m) for w in ("gpt2_small", "tinyllama") for m in ("pad_to_max", "dynamic_pad", "length_bucketed", "pack")
    }
    for entry in discover_experiment_configs(CONFIG_ROOT):
        assert scaling_issues(entry.config) == []
        assert entry.config.hardware.gpus == 2
    conditions = load_platform_conditions("platforms/execution_matrix.json", "l40s", wave=CANARY_WAVE)
    assert len(conditions) == 8 and all(c.required_trials == 1 for c in conditions)
    path, text = expected_matrix()
    assert path.read_text() == text
    perf = load_platform_conditions("platforms/execution_matrix.json", "l40s", wave="scaling-dd-real-text")
    assert len(perf) == 32 and sum(c.required_trials for c in perf) == 96


def test_s7_performance_uses_focused_gate_before_bundle_creation(monkeypatch):
    def reject(*args, **kwargs):
        assert kwargs["canary_wave"] == CANARY_WAVE
        raise ValueError("S7 not collected")

    monkeypatch.setattr(gate, "verify_deep_dive_canaries", reject)
    with pytest.raises(ValueError, match="S7 not collected"):
        _submit_platform_wave(
            SimpleNamespace(
                wave="scaling-dd-real-text",
                platform_profile="platforms/l40s.json",
                venv_activate="/recovered/activate",
                execution_matrix="platforms/execution_matrix.json",
            )
        )


def test_all_eight_real_text_modes_are_required_and_accounted(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "validate_result_file", lambda path: [])
    monkeypatch.setattr(gate, "_canary_issues", lambda *args: [])
    entries = {e.matrix_id: e for e in discover_experiment_configs(CONFIG_ROOT)}
    profile = load_platform_profile("platforms/l40s.json")
    manifest = []
    for condition in load_platform_conditions("platforms/execution_matrix.json", "l40s", wave=CANARY_WAVE):
        c = resolve_platform_config(entries[condition.matrix_id], profile, condition)
        nominal = c.runtime.max_steps * c.workload.parameters["global_batch_size"] * 1024
        processed = nominal
        useful = nominal if c.workload.parameters["text_batching"] == "pack" else nominal // 2
        r = dict(
            config=c.to_dict(),
            environment=dict(
                torch_version="2.8.0",
                cuda_version="12.4",
                nccl_version="(2, 27, 3)",
                cudnn_version="8907",
                transformers_version="4.57.6",
            ),
            metrics=dict(
                total_tokens=useful,
                processed_tokens=processed,
                nominal_tokens=nominal,
                measured_seconds=10,
                throughput_tokens_sec=useful / 10,
                processed_tokens_sec=processed / 10,
                padding_fraction=1 - useful / processed,
                rank_token_imbalance=1,
                text_batching=c.workload.parameters["text_batching"],
                step_time_p50_ms=10,
                step_time_p95_ms=11,
            ),
        )
        assert gate._real_text_issues(r, c) == []
        path = tmp_path / c.experiment_id / "trial.json"
        path.parent.mkdir()
        path.write_text(json.dumps(r))
        manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(tmp_path).as_posix()}\n")
    (tmp_path / "SHA256SUMS").write_text("".join(manifest))
    assert len(gate.verify_deep_dive_canaries(tmp_path, "l40s", canary_wave=CANARY_WAVE)) == 8
    r["metrics"]["padding_fraction"] = 0.5  # packed streams must have no padding
    assert "packed token accounting" in gate._real_text_issues(r, c)
    (tmp_path / "E810" / "trial.json").unlink()
    with pytest.raises(ValueError, match="E810"):
        gate.verify_deep_dive_canaries(tmp_path, "l40s", canary_wave=CANARY_WAVE)
