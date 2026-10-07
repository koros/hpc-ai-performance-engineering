import csv
import json
from pathlib import Path

from hpc_ai_perf.config import load_experiment_config
from hpc_ai_perf.results import ExperimentResult, aggregate_results
from hpc_ai_perf.results.contract import classify_outcome
from hpc_ai_perf.validation.project import validate_result_file


def test_new_result_records_preserve_samples_and_legacy_status(tmp_path: Path) -> None:
    config = load_experiment_config(Path("configs/inference/E702_tinyllama_tp2_long_prompt.json"))
    result = ExperimentResult.from_config(
        config,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:02Z",
        runtime_seconds=2.0,
        metrics={"throughput_tokens_sec": 100.0},
        samples={"latency_seconds": [0.10, 0.20, 0.30]},
    )

    path = result.write_json(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.4.0"
    assert payload["status"] == "completed"
    assert payload["outcome"] == "success"
    assert payload["samples"]["latency_seconds"] == [0.10, 0.20, 0.30]

    output_path = aggregate_results(tmp_path, tmp_path / "processed.csv")
    with output_path.open(encoding="utf-8", newline="") as file_obj:
        row = next(csv.DictReader(file_obj))
    assert row["outcome"] == "success"
    assert row["sample_count_latency_seconds"] == "3"


def test_failure_classification_preserves_legacy_failed_status() -> None:
    assert classify_outcome("completed") == "success"
    assert classify_outcome("failed", "CUDA out of memory") == "oom"
    assert classify_outcome("failed", "vLLM is not installed") == "unsupported"
    assert classify_outcome("failed", "unexpected scheduler interruption") == "failed"


def test_validation_accepts_structured_samples_and_rejects_nested_samples(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "1.4.0",
                "experiment_id": "SMOKE",
                "experiment": "contract smoke",
                "workload": "synthetic",
                "model": "synthetic",
                "mode": "inference",
                "gpus": 0,
                "precision": "fp32",
                "status": "failed",
                "outcome": "unsupported",
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T00:00:01Z",
                "runtime_seconds": 1.0,
                "metrics": {},
                "samples": {"latency_seconds": [0.1, 0.2]},
                "environment": {},
                "config": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate_result_file(result_path) == []

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(
        result_path.read_text(encoding="utf-8").replace("[0.1, 0.2]", '[{"nested": true}]'),
        encoding="utf-8",
    )
    issues = validate_result_file(invalid_path)
    assert any("samples must map names to arrays" in issue.message for issue in issues)
