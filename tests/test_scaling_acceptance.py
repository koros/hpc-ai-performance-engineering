"""Exercise the measured, immutable evidence gate before spending the main wave."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from hpc_ai_perf.cli import _submit_platform_wave
from hpc_ai_perf.config import config_sha256, discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.results import ExperimentResult
from hpc_ai_perf.results.evidence import write_evidence_manifest
from hpc_ai_perf.validation.scaling_acceptance import CANARY_IDS, verify_scaling_canaries
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256


def collected_canaries(tmp_path: Path, platform="h100") -> Path:
    root = tmp_path / "raw" / platform
    entries = {entry.config.experiment_id: entry for entry in discover_experiment_configs("configs/scaling_studies")}
    profile = load_platform_profile(f"platforms/{platform}.json")
    for condition in load_platform_conditions("platforms/execution_matrix.json", platform, wave="scaling-batch-canary"):
        eid = condition.matrix_id
        cfg = resolve_platform_config(entries[eid], profile, condition)
        params = cfg.workload.parameters
        gpus = cfg.hardware.gpus
        updates = cfg.runtime.max_steps + params["warmup_steps"]
        metrics = {
            "steps": cfg.runtime.max_steps,
            "warmup_steps": params["warmup_steps"],
            "measured_seconds": 10.0,
            "gpu_hours": 10.0 * gpus / 3600,
            "total_tokens": cfg.runtime.max_steps * params["global_batch_size"] * params["sequence_length"],
            "total_samples": cfg.runtime.max_steps * params["global_batch_size"],
            "throughput_tokens_sec": 100.0,
            "throughput_unit": "training_tokens_per_second",
            "loss": 1.0,
            "initial_training_loss": 1.1,
            "memory_used_gb": 20.0,
            "world_size": gpus,
            "batch_size_per_gpu": params["per_gpu_batch_size"],
            "effective_global_batch_size": params["global_batch_size"],
            "sequence_length": params["sequence_length"],
            "optimizer_step_calls": updates,
            "backward_calls": updates * params["gradient_accumulation_steps"],
            "input_micro_batches_consumed": updates * params["gradient_accumulation_steps"],
            "compute_dtype_observed": "bf16",
            "parameter_dtype_observed": "torch.float32",
            "strategy_observed": "ddp",
            "attention_backend_observed": "sdpa",
            "training_use_cache": False,
            "dataset_execution": params["input_protocol"],
            "global_input_sha256": "a" * 64 if eid != "E317" else "b" * 64,
            "scaling_runtime_sha256": scaling_runtime_sha256(),
            "training_rank_placement": json.dumps([{"rank": index} for index in range(gpus)]),
            "nvidia_smi.profiled_gpu_count": gpus,
            "nvidia_smi.measurement_status": "sufficient",
            "nvidia_smi.energy_joules_measured_region": 900.0,
        }
        environment = {
            "observed_gpu_model": "NVIDIA H100 80GB HBM3" if platform == "h100" else "NVIDIA L40S",
            "git_commit": "a" * 40,
            "git_dirty": False,
            "config_sha256": config_sha256(cfg),
            "command": "python -m hpc_ai_perf.cli run",
        }
        result = ExperimentResult.from_config(
            cfg,
            status="completed",
            started_at="2026-09-30T00:00:00Z",
            completed_at="2026-09-30T00:00:10Z",
            runtime_seconds=10.0,
            metrics=metrics,
            environment=environment,
        )
        result.write_json(root, preserve_trial=True)
    write_evidence_manifest(root)
    return root


@pytest.mark.parametrize("platform", ["h100", "l40s"])
def test_acceptance_requires_platform_specific_canaries(platform, tmp_path):
    root = collected_canaries(tmp_path, platform)
    accepted = verify_scaling_canaries(root, platform)
    assert set(accepted) == CANARY_IDS
    assert {accepted[eid]["input_sha256"] for eid in ("E314", "E315", "E316")} == {"a" * 64}


def test_acceptance_rejects_absent_or_tampered_collected_evidence(tmp_path):
    root = collected_canaries(tmp_path)
    (root / "E314").rename(root / "missing")
    with pytest.raises(ValueError, match="E314: no completed"):
        verify_scaling_canaries(root, "h100")
    (root / "missing").rename(root / "E314")
    record = next((root / "E314").glob("*.json"))
    record.write_text(record.read_text().replace('"loss": 1.0', '"loss": 2.0'))
    with pytest.raises(ValueError, match="SHA256SUMS mismatch"):
        verify_scaling_canaries(root, "h100")


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("optimizer_step_calls", 1, "optimizer updates"),
        ("compute_dtype_observed", "fp32", "BF16 compute"),
        ("nvidia_smi.measurement_status", "insufficient_samples", "measured energy"),
        ("global_input_sha256", "c" * 64, "input digests differ"),
    ],
)
def test_acceptance_rejects_failed_control_even_if_checksum_updated(tmp_path, key, value, message):
    root = collected_canaries(tmp_path)
    record = next((root / "E314").glob("*.json"))
    payload = json.loads(record.read_text())
    payload["metrics"][key] = value
    record.write_text(json.dumps(payload))
    write_evidence_manifest(root)
    with pytest.raises(ValueError, match=message):
        verify_scaling_canaries(root, "h100")


def test_main_wave_cli_checks_acceptance_before_creating_bundle(tmp_path, monkeypatch):
    from hpc_ai_perf.validation import scaling_acceptance

    def reject(root, platform, **kwargs):
        assert root == Path("results/raw/h100")
        assert platform == "h100"
        raise ValueError("acceptance denied")

    monkeypatch.setattr(scaling_acceptance, "verify_scaling_canaries", reject)
    with pytest.raises(ValueError, match="acceptance denied"):
        _submit_platform_wave(
            Namespace(
                wave="scaling-batch-crossover",
                platform_profile=str(Path(__file__).resolve().parents[1] / "platforms/h100.json"),
                execution_matrix=str(Path(__file__).resolve().parents[1] / "platforms/execution_matrix.json"),
            )
        )
