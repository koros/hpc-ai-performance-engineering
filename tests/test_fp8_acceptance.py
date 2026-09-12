from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_fp8_acceptance_verifier_requires_both_model_paths(tmp_path: Path) -> None:
    for experiment_id, workload in (
        ("fp8_gpt2_training_acceptance", "gpt2_small"),
        ("fp8_distilbert_training_acceptance", "distilbert"),
    ):
        record = {
            "experiment_id": experiment_id,
            "workload": workload,
            "precision": "fp8",
            "status": "completed",
            "error": None,
            "metrics": {
                "precision_backend": "transformer_engine_delayed_scaling_hybrid",
                "compute_dtype_observed": "fp8_mixed_bf16",
                "fp8_forward_enabled": True,
                "transformer_engine_module_count": 4,
                "transformer_engine_version": "2.8.0",
                "throughput_tokens_sec": 100.0,
                "loss": 1.0,
                "quality_finite": True,
                "quality_loss": 1.1,
            },
            "environment": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "observed_gpu_model": "NVIDIA L40S",
            },
        }
        (tmp_path / f"{experiment_id}.json").write_text(json.dumps(record), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_fp8_training_acceptance.py",
            "--result-root",
            str(tmp_path),
            "--expected-gpu-model",
            "L40S",
            "--expected-git-commit",
            "a" * 40,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["status"] == "completed"
    assert set(summary["results"]) == {
        "fp8_gpt2_training_acceptance",
        "fp8_distilbert_training_acceptance",
    }


def test_fp8_acceptance_verifier_rejects_a_non_fp8_backend(tmp_path: Path) -> None:
    record = {
        "experiment_id": "fp8_gpt2_training_acceptance",
        "workload": "gpt2_small",
        "precision": "fp8",
        "status": "completed",
        "metrics": {"precision_backend": "pytorch_amp_bf16"},
        "environment": {"git_commit": "a" * 40, "git_dirty": False, "observed_gpu_model": "NVIDIA L40S"},
    }
    (tmp_path / "fp8_gpt2_training_acceptance.json").write_text(json.dumps(record), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_fp8_training_acceptance.py",
            "--result-root",
            str(tmp_path),
            "--expected-gpu-model",
            "L40S",
            "--expected-git-commit",
            "a" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Transformer Engine precision backend was not observed" in completed.stderr
    assert "missing result" in completed.stderr
