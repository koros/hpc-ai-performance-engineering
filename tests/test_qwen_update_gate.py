import json
from pathlib import Path

import pytest

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.validation.qwen7b_training import update_runtime_sha256, verify_update_canaries


def write_canaries(root):
    entries = {e.config.experiment_id: e for e in discover_experiment_configs("configs/large_model_training")}
    profile = load_platform_profile("platforms/h100.json")
    for condition in load_platform_conditions("platforms/execution_matrix.json", "h100", wave="qwen7b-update-canary"):
        cfg = resolve_platform_config(entries[condition.matrix_id], profile, condition)
        updates = cfg.runtime.max_steps + cfg.workload.parameters["warmup_steps"]
        path = root / condition.matrix_id
        path.mkdir(exist_ok=True)
        path.joinpath("trial.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "error": None,
                    "condition_id": condition.condition_id,
                    "gpus": cfg.hardware.gpus,
                    "config": cfg.to_dict(),
                    "environment": {"observed_gpu_model": "NVIDIA H100", "git_dirty": False},
                    "metrics": {
                        "training_update_runtime_sha256": update_runtime_sha256(),
                        "strategy_observed": cfg.distributed.strategy,
                        "initial_training_loss": 11,
                        "loss": 10,
                        "quality_loss": 12,
                        "quality_finite": True,
                        "throughput_tokens_sec": 100,
                        "optimizer_step_calls": updates,
                        "backward_calls": updates * 8,
                        "optimizer_state_dtypes_observed": "torch.float32",
                        "effective_global_batch_size": 16,
                        "deepspeed_accumulation_mode": "explicit_boundary_v1",
                        "deepspeed_optimizer_updates": updates,
                        "deepspeed_samples_accounted": updates * 16,
                    },
                }
            )
        )


def test_update_gate_accepts_current_accounting(tmp_path):
    write_canaries(tmp_path)
    assert set(verify_update_canaries(tmp_path)) == {"E262", "E263"}


def test_update_gate_requires_both_sharded_canaries(tmp_path):
    write_canaries(tmp_path)
    (tmp_path / "E262" / "trial.json").unlink()
    with pytest.raises(ValueError, match="E262: no collected record matches"):
        verify_update_canaries(tmp_path)


@pytest.mark.parametrize("experiment_id", ["E261", "E264", "E267"])
def test_update_gate_cannot_release_unvalidated_ddp_via_custom_policy(tmp_path, experiment_id):
    write_canaries(tmp_path)
    policy = json.loads(Path("platforms/execution_matrix.json").read_text())
    for entry in policy["deferred"]:
        if experiment_id in entry["experiment_ids"]:
            entry["experiment_ids"].remove(experiment_id)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Qwen (update canary policy|DDP controls)"):
        verify_update_canaries(tmp_path, execution_matrix=matrix)


@pytest.mark.parametrize(
    "field,value",
    [
        ("deepspeed_optimizer_updates", 1),
        ("deepspeed_samples_accounted", 1),
        ("training_update_runtime_sha256", "stale"),
        ("optimizer_state_dtypes_observed", "unavailable"),
        ("initial_training_loss", 20),
    ],
)
def test_update_gate_rejects_invalid_accounting_and_changed_runtime(tmp_path, field, value):
    write_canaries(tmp_path)
    path = tmp_path / "E263" / "trial.json"
    record = json.loads(path.read_text())
    record["metrics"][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Qwen training update canary failed"):
        verify_update_canaries(tmp_path)


def test_new_update_waves_are_h100_only_and_keep_old_configs_unchanged():
    conditions = load_platform_conditions("platforms/execution_matrix.json", "h100")
    entries = {e.config.experiment_id: e.config for e in discover_experiment_configs("configs/large_model_training")}
    assert "deepspeed_accumulation_mode" not in entries["E253"].workload.parameters
    for c in conditions:
        if c.wave.startswith("qwen7b-update"):
            cfg = entries[c.matrix_id]
            assert cfg.hardware.gpus * cfg.workload.parameters["gradient_accumulation_steps"] == 16
            assert cfg.workload.parameters["training_update_diagnostics"] is True
    assert not load_platform_conditions("platforms/execution_matrix.json", "l40s", wave="qwen7b-update-canary")
