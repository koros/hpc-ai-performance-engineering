import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.generate_topology_recovery import expected_files

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.config.scaling import scaling_issues
from hpc_ai_perf.config.topology_recovery import CANARY_WAVE, CONFIG_ROOT, PERFORMANCE_WAVE, design
from hpc_ai_perf.validation import capability_issues
from hpc_ai_perf.validation import topology_recovery as gate
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256


def test_recovery_keeps_historical_configs_and_scopes_algorithms():
    from hpc_ai_perf.config.topology_recovery import PERFORMANCE_SOURCES
    from hpc_ai_perf.slurm.job import SlurmJob

    rows = design()
    assert [eid for eid, _, _ in rows] == [f"E{i}" for i in range(791, 810)]
    assert len([w for _, w, _ in rows if w == CANARY_WAVE]) == 5
    assert len([w for _, w, _ in rows if w == PERFORMANCE_WAVE]) == 14
    assert set(PERFORMANCE_SOURCES) == {
        "E389",
        "E616",
        "E617",
        "E623",
        "E625",
        "E626",
        "E627",
        "E628",
        "E629",
        "E630",
        "E421",
        "E631",
        "E632",
        "E633",
    }
    for entry in discover_experiment_configs(CONFIG_ROOT):
        c = entry.config
        assert scaling_issues(c) == [] and capability_issues(c) == []
        assert c.workload.parameters["gradient_accumulation_steps"] == 1
        assert c.workload.parameters["scaling_type"] == "weak"
        algo = c.runtime.environment.get("NCCL_ALGO")
        assert algo in (None, "allreduce:Tree", "allreduce:Ring")
        if algo:
            assert f"export NCCL_ALGO={algo}" in SlurmJob.from_config(c, entry.path).render()
        if c.metadata.get("allocation_gpus"):
            assert SlurmJob.from_config(c, entry.path).gpus_per_node == 8
    from pathlib import Path

    legacy = json.loads(next(Path("configs/scaling_deep_dive").glob("E617_*.json")).read_text())
    assert legacy["runtime"]["environment"]["NCCL_ALGO"] == "Tree"


def test_generated_files_match_design():
    for path, text in expected_files().items():
        assert path.read_text() == text
    assert len(discover_experiment_configs(CONFIG_ROOT)) == 19


def test_performance_requires_collected_canaries_before_submission(monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("acceptance not collected")

    monkeypatch.setattr(gate, "verify_topology_canaries", reject)
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
                nccl_version="(2, 27, 3)",
                cudnn_version="8907",
            ),
            metrics=dict(
                gpu_topology_matrix="GPU0 GPU1 CPU Affinity",
                gpu_pcie_link_inventory="GPU-0, 0000:01:00.0, 4, 4, 16, 16",
                training_rank_placement=json.dumps(
                    [
                        dict(
                            rank=i,
                            gpu_uuid=f"GPU-{i}",
                            cuda_visible_devices=c.runtime.environment.get("CUDA_VISIBLE_DEVICES", ""),
                        )
                        for i in range(c.hardware.gpus)
                    ]
                ),
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
    assert len(gate.verify_topology_canaries(collected)) == 5


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
    path = collected / ("E794" if reason == "global inputs differ" else "E791") / "trial.json"
    record = json.loads(path.read_text())
    mutation(record)
    path.write_text(json.dumps(record))
    checksum(collected)
    with pytest.raises(ValueError, match=reason):
        gate.verify_topology_canaries(collected)


def test_new_failed_attempt_cannot_fall_back_to_previous_pass(collected):
    r = json.loads((collected / "E791" / "trial.json").read_text())
    r.update(status="failed", started_at="2026-10-23T00:00:00Z")
    (collected / "E791" / "retry.json").write_text(json.dumps(r))
    checksum(collected)
    with pytest.raises(ValueError, match="completed"):
        gate.verify_topology_canaries(collected)


def test_checksum_tampering_fails(collected):
    path = collected / "E791" / "trial.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        gate.verify_topology_canaries(collected)


def test_old_nccl_is_rejected(collected):
    path = collected / "E791" / "trial.json"
    record = json.loads(path.read_text())
    record["environment"]["nccl_version"] = "(2, 23, 4)"
    path.write_text(json.dumps(record))
    checksum(collected)
    with pytest.raises(ValueError, match="NCCL 2.24"):
        gate.verify_topology_canaries(collected)


def test_audit_reports_compatibility_without_changing_sources(tmp_path, monkeypatch):
    from scripts import audit_topology_recovery as module
    from scripts.audit_topology_recovery import audit

    monkeypatch.setattr(module, "validate_result_file", lambda path: [])
    path = tmp_path / "results/E626/run.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(status="completed", metrics=dict(scaling_runtime_sha256="old"), environment={})))
    original = path.read_bytes()
    report = audit(tmp_path, tmp_path / "no-current-evidence")
    assert report["historical_results"] == {"completed": 1}
    assert report["historical_fingerprint_matches_current"] is False
    assert report["current_comparison_evidence"] == []
    assert path.read_bytes() == original
