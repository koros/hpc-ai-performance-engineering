from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpc_ai_perf.config import config_sha256
from hpc_ai_perf.validation import qwen7b_training
from hpc_ai_perf.workloads import transformer as transformer_workload


def _record(experiment_id: str, config_hash: str, strategy: str, detail: str) -> dict[str, object]:
    return {
        "status": "completed",
        "error": None,
        "condition_id": f"{experiment_id}@h100",
        "workload": "qwen2_5_7b_training",
        "precision": "bf16",
        "gpus": 2,
        "config_sha256": config_hash,
        "environment": {"observed_gpu_model": "NVIDIA H100 80GB HBM3", "git_dirty": False},
        "metrics": {
            "strategy_observed": strategy,
            "strategy_detail_observed": detail,
            "model_load_dtype_requested": "bf16",
            "effective_global_batch_size": 16,
            "sequence_length": 512,
            "activation_checkpointing": True,
            "throughput_tokens_sec": 100.0,
            "loss": 1.0,
            "quality_finite": True,
            "quality_loss": 1.1,
            "memory_used_gb": 20.0,
            **(
                {
                    "fsdp_auto_wrap_policy": "qwen_decoder_layer",
                    "fsdp_limit_all_gathers": True,
                    "fsdp_forward_prefetch": False,
                }
                if experiment_id == "E250"
                else {}
            ),
        },
    }


def test_qwen_fsdp_decoder_layer_is_derived_from_the_loaded_model() -> None:
    class Qwen2DecoderLayer:
        pass

    class Model:
        class Core:
            layers = [Qwen2DecoderLayer()]

        model = Core()

    assert transformer_workload._qwen_decoder_layer_class(Model()) is Qwen2DecoderLayer


def test_verify_qwen7b_canaries_accepts_current_controlled_records(tmp_path: Path) -> None:
    expected = qwen7b_training.resolved_canary_configs("h100")
    for experiment_id, expectation in qwen7b_training.EXPECTATIONS.items():
        root = tmp_path / experiment_id
        root.mkdir()
        root.joinpath("trial.json").write_text(
            json.dumps(
                _record(
                    experiment_id,
                    config_sha256(expected[experiment_id]),
                    expectation["strategy"],
                    expectation["detail"],
                )
            ),
            encoding="utf-8",
        )

    accepted = qwen7b_training.verify_canaries(tmp_path)

    assert set(accepted) == {"E250", "E251"}


def test_verify_qwen7b_canaries_rejects_wrong_runtime_strategy(tmp_path: Path) -> None:
    expected = qwen7b_training.resolved_canary_configs("h100")
    for experiment_id, expectation in qwen7b_training.EXPECTATIONS.items():
        root = tmp_path / experiment_id
        root.mkdir()
        strategy = "ddp" if experiment_id == "E250" else expectation["strategy"]
        root.joinpath("trial.json").write_text(
            json.dumps(
                _record(
                    experiment_id,
                    config_sha256(expected[experiment_id]),
                    strategy,
                    expectation["detail"],
                )
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="E250: strategy failed"):
        qwen7b_training.verify_canaries(tmp_path)
