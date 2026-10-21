import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.generate_microbatch_accumulation import expected_files

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.microbatch_accumulation import CANARY_WAVE, CONFIG_ROOT, PERFORMANCE_WAVE, design
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.config.scaling import scaling_issues
from hpc_ai_perf.config.scaling_deep_dive import CAPACITY_GB, MEMORY_RULE, Condition, estimated_memory_gb
from hpc_ai_perf.validation import capability_issues
from hpc_ai_perf.validation import microbatch_accumulation as gate
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256


def test_design_is_matched_weak_scaling_and_respects_memory_budget():
    rows = design()
    assert [eid for eid, _, _ in rows] == [f"E{i}" for i in range(773, 791)]
    assert sum(w == CANARY_WAVE for _, w, _ in rows) == 6
    performance = [d for _, w, d in rows if w == PERFORMANCE_WAVE]
    assert len(performance) == 12
    assert {(d["hardware"]["gpus"], d["workload"]["parameters"]["per_gpu_batch_size"]) for d in performance} == {
        (g, b) for g in (1, 2, 4, 8) for b in (1, 2, 4)
    }
    for entry in discover_experiment_configs(CONFIG_ROOT):
        c = entry.config
        p = c.workload.parameters
        assert scaling_issues(c) == [] and capability_issues(c) == []
        assert p["per_gpu_batch_size"] * p["gradient_accumulation_steps"] == 16
        assert p["global_batch_size"] == c.hardware.gpus * 16
        assert p["scaling_type"] == "weak" and p["fixed_workload"] is False
        assert p["sequence_length"] == 1024
        estimate = estimated_memory_gb(Condition("tinyllama", c.hardware.gpus, p["per_gpu_batch_size"], 1024))
        assert estimate < CAPACITY_GB["l40s"] * MEMORY_RULE
        assert c.metadata["study_design_note"] == "follow-up motivated by observed results"
    assert estimated_memory_gb(Condition("tinyllama", 8, 8, 1024)) > CAPACITY_GB["l40s"] * MEMORY_RULE


def test_generated_files_match_design():
    for path, text in expected_files().items():
        assert path.read_text() == text
    assert len(discover_experiment_configs(CONFIG_ROOT)) == 18


def test_performance_requires_collected_canaries_before_submission(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("acceptance not collected")

    monkeypatch.setattr(gate, "verify_microbatch_canaries", reject)
    with pytest.raises(ValueError, match="acceptance not collected"):
        _submit_platform_wave(
            SimpleNamespace(
                wave=PERFORMANCE_WAVE,
                platform_profile="platforms/l40s.json",
                venv_activate="/recovered/activate",
                execution_matrix="platforms/execution_matrix.json",
            )
        )


@pytest.fixture
def collected(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "validate_result_file", lambda path: [])
    entries = {e.matrix_id: e for e in discover_experiment_configs(CONFIG_ROOT)}
    profile = load_platform_profile("platforms/l40s.json")
    for condition in load_platform_conditions("platforms/execution_matrix.json", "l40s", wave=CANARY_WAVE):
        c = resolve_platform_config(entries[condition.matrix_id], profile, condition)
        p = c.workload.parameters
        updates = p["warmup_steps"] + c.runtime.max_steps
        record = dict(
            status="completed",
            error=None,
            config=c.to_dict(),
            started_at="2026-10-22T00:00:00Z",
            platform_id="l40s",
            condition_id=condition.condition_id,
            gpus=c.hardware.gpus,
            git_dirty=False,
            environment=dict(
                git_dirty=False,
                observed_gpu_model="NVIDIA L40S",
                torch_version="2.8.0",
                cuda_version="12.4",
                transformers_version="4.57.6",
            ),
            metrics=dict(
                world_size=c.hardware.gpus,
                scaling_runtime_sha256=scaling_runtime_sha256(),
                dataset_execution=p["input_protocol"],
                global_input_sha256="a" * 64,
                optimizer_step_calls=updates,
                expected_optimizer_step_calls=updates,
                backward_calls=updates * p["gradient_accumulation_steps"],
                effective_global_batch_size=p["global_batch_size"],
                strategy_observed="ddp",
                compute_dtype_observed="bf16",
                attention_backend_observed="sdpa",
                training_use_cache=False,
                throughput_tokens_sec=100,
                initial_training_loss=10,
                loss=9,
                memory_used_gb=35,
                ddp_comm_hook_observed="none",
            ),
        )
        path = tmp_path / c.experiment_id / "trial.json"
        path.parent.mkdir()
        path.write_text(json.dumps(record))
    checksum(tmp_path)
    return tmp_path


def checksum(root):
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n"
            for p in sorted(root.glob("E*/*.json"))
        )
    )


def test_current_matching_evidence_unlocks_performance(collected):
    assert len(gate.verify_microbatch_canaries(collected)) == 6


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r["metrics"].update(scaling_runtime_sha256="0" * 64), "runtime fingerprint"),
        (lambda r: r["metrics"].update(backward_calls=1), "update accounting"),
        (lambda r: r["metrics"].update(memory_used_gb=44), "memory ceiling"),
        (lambda r: r["metrics"].update(global_input_sha256="b" * 64), "global inputs differ"),
        (lambda r: r["environment"].update(torch_version="2.9.0"), "runtime versions differ"),
        (lambda r: r.update(status="failed"), "completed"),
        (lambda r: r["config"]["workload"]["parameters"].update(per_gpu_batch_size=2), "no matching"),
    ],
)
def test_invalid_evidence_does_not_unlock_wave(collected, mutation, reason):
    path = collected / "E773" / "trial.json"
    record = json.loads(path.read_text())
    mutation(record)
    path.write_text(json.dumps(record))
    checksum(collected)
    with pytest.raises(ValueError, match=reason):
        gate.verify_microbatch_canaries(collected)


def test_new_failed_attempt_cannot_fall_back_to_previous_pass(collected):
    r = json.loads((collected / "E773" / "trial.json").read_text())
    r.update(status="failed", started_at="2026-10-23T00:00:00Z")
    (collected / "E773" / "retry.json").write_text(json.dumps(r))
    checksum(collected)
    with pytest.raises(ValueError, match="completed"):
        gate.verify_microbatch_canaries(collected)


def test_checksum_tampering_fails(collected):
    path = collected / "E773" / "trial.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        gate.verify_microbatch_canaries(collected)
