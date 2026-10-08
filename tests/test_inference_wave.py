import pytest

from hpc_ai_perf.analysis.summaries import (
    build_inference_tp_crossover_summary,
    build_inference_tp_precision_summary,
)
from hpc_ai_perf.workloads.transformer import (
    _decode_timing_metrics,
    _decode_timing_samples,
    _serving_tail_metrics,
)


def _row(tp: int, prompt: int = 1920, throughput: float = 100.0, latency: float = 0.2) -> dict[str, str]:
    return {
        "status": "completed",
        "mode": "inference",
        "condition_id": f"E{700 + tp}@h100",
        "experiment_id": f"E{700 + tp}",
        "platform_id": "h100",
        "comparability": "exact",
        "workload": "tinyllama",
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "precision": "bf16",
        "framework": "vllm",
        "gpus": str(tp),
        "batch_size": "1",
        "throughput_unit": "generated_tokens_per_second",
        "parameter_tensor_parallel_size": str(tp),
        "parameter_prompt_length": str(prompt),
        "parameter_max_new_tokens": "128",
        "parameter_concurrent_requests": "1",
        "parameter_serving_layout": f"tp{tp}",
        "throughput_tokens_sec": str(throughput),
        "latency_p95_ms": str(latency * 1000),
        "latency_p99_ms": str(latency * 1100),
        "time_to_first_token_p95_ms": str(latency * 500),
        "goodput_requests_sec": "10",
    }


def test_tp_crossover_keeps_prompt_lengths_separate_and_reports_relative_change() -> None:
    rows = [_row(1), _row(2, throughput=180.0, latency=0.25), _row(4, throughput=160.0, latency=0.3)]
    rows.append(_row(1, prompt=128, throughput=200.0, latency=0.1))

    summaries = build_inference_tp_crossover_summary(rows)

    long_prompt = [row for row in summaries if row["prompt_length"] == "1920"]
    assert [row["tp_degree"] for row in long_prompt] == [1, 2, 4]
    assert long_prompt[0]["throughput_change_pct_vs_baseline"] == 0.0
    assert long_prompt[1]["throughput_change_pct_vs_baseline"] == 80.0
    assert long_prompt[2]["latency_p95_change_pct_vs_baseline"] == 50.0
    assert len([row for row in summaries if row["prompt_length"] == "128"]) == 1


def test_tp_crossover_does_not_pool_replica_layout_with_single_engine_tp() -> None:
    replica = _row(2, throughput=240.0, latency=0.4)
    replica.update(
        {
            "condition_id": "E708@h100",
            "gpus": "4",
            "parameter_serving_layout": "tp2_dp2",
            "parameter_data_parallel_size": "2",
            "parameter_replica_count": "2",
            "parameter_request_routing": "round_robin",
        }
    )

    summaries = build_inference_tp_crossover_summary([_row(1), _row(2), replica])

    tp2_rows = [row for row in summaries if row["tp_degree"] == 2]
    assert {row["serving_layout"] for row in tp2_rows} == {"tp2", "tp2_dp2"}
    assert {row["trials"] for row in tp2_rows} == {1}
    replica_row = next(row for row in tp2_rows if row["serving_layout"] == "tp2_dp2")
    assert replica_row["data_parallel_size"] == "2"
    assert replica_row["request_routing"] == "round_robin"


def test_tp_precision_summary_compares_fp8_with_matched_bf16_control() -> None:
    bf16 = _row(4, prompt=1280, throughput=100.0, latency=0.4)
    bf16.update(
        {
            "condition_id": "E710@h100",
            "experiment_id": "E710",
            "phase": "Inference TP precision crossover",
            "parameter_concurrent_requests": "4",
            "parameter_serving_layout": "tp4",
            "precision": "bf16",
        }
    )
    fp8 = dict(bf16)
    fp8.update(
        {
            "condition_id": "E712@h100",
            "experiment_id": "E712",
            "precision": "fp8",
            "throughput_tokens_sec": "125",
            "latency_p95_ms": "500",
            "latency_p99_ms": "550",
            "quality_request_success_rate": "1.0",
        }
    )

    summaries = build_inference_tp_precision_summary([bf16, fp8])

    assert {row["precision"] for row in summaries} == {"bf16", "fp8"}
    fp8_summary = next(row for row in summaries if row["precision"] == "fp8")
    assert fp8_summary["baseline_precision"] == "bf16"
    assert fp8_summary["throughput_change_pct_vs_baseline"] == 25.0
    assert fp8_summary["latency_p95_change_pct_vs_baseline"] == 25.0


def test_serving_tail_metrics_reports_p99_and_explicit_goodput() -> None:
    metrics = _serving_tail_metrics(
        [0.10, 0.20, 0.30, 0.40],
        [0.05, 0.06, 0.07, 0.08],
        [0.01, 0.02, 0.03, 0.04],
        1.0,
        250,
    )

    assert metrics["latency_p99_ms"] == pytest.approx(397.0)
    assert metrics["time_to_first_token_p99_ms"] == pytest.approx(79.7)
    assert metrics["queueing_time_p99_ms"] == pytest.approx(39.7)
    assert metrics["goodput_requests_sec"] == 2.0
    assert metrics["goodput_slo_met_fraction"] == 0.5


def test_decode_timing_uses_backend_last_token_timestamp() -> None:
    metrics = type("Metrics", (), {"first_token_time": 10.0, "last_token_time": 13.0})()

    tpot, decode_time = _decode_timing_samples(metrics, 4)

    assert tpot == pytest.approx(1.0)
    assert decode_time == pytest.approx(3.0)
    summary = _decode_timing_metrics([tpot], [decode_time])
    assert summary["decode_timing_available"] is True
    assert summary["decode_timing_source"] == "vllm_last_token_time"
    assert summary["time_per_output_token_p95_ms"] == pytest.approx(1_000.0)
    assert summary["decode_time_p95_ms"] == pytest.approx(3_000.0)


def test_decode_timing_is_unavailable_without_last_token_timestamp() -> None:
    metrics = type("Metrics", (), {"first_token_time": 10.0})()

    assert _decode_timing_samples(metrics, 4) == (None, None)
    summary = _decode_timing_metrics([], [])
    assert summary["decode_timing_available"] is False
    assert summary["decode_timing_source"] is None
