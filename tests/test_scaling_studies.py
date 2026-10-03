from copy import deepcopy

import pytest

from hpc_ai_perf.analysis.summaries import build_scaling_summary


def row(gpus=1, batch=8, **changes):
    return {
        "status": "completed",
        "mode": "training",
        "experiment_id": f"test-{gpus}-{batch}",
        "platform_id": "h100",
        "comparability": "exact",
        "phase": "Strong scaling",
        "scaling_type": "strong",
        "model": "TinyLlama",
        "workload": "tinyllama",
        "precision": "bf16",
        "strategy": "ddp",
        "gpus": str(gpus),
        "global_batch_size": str(batch),
        "per_gpu_batch_size": str(batch // gpus),
        "metric_measured_seconds": str(12 / gpus),
        "throughput_tokens_sec": str(batch * 512 * gpus / 12),
        "throughput_unit": "training_tokens_per_second",
        "parameter_sequence_length": "512",
        "parameter_gradient_accumulation_steps": "1",
        **changes,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("global_batch_size", "16"),
        ("parameter_sequence_length", "1024"),
        ("parameter_activation_checkpointing", "True"),
        ("dataset", "real-text"),
        ("parameter_gradient_accumulation_steps", "2"),
        ("model_revision", "other"),
        ("metric_parameter_dtype_observed", "torch.bfloat16"),
        ("metric_scaling_runtime_sha256", "other"),
        ("environment_torch_version", "other"),
        ("parameter_learning_rate", "0.01"),
        ("metric_global_input_sha256", "different-samples"),
        ("measured_iterations", "200"),
        ("runtime_environment", '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}'),
    ],
)
def test_scaling_does_not_pool_changed_fixed_settings(field, value):
    baseline, treatment = row(), row(2)
    treatment[field] = value
    summaries = build_scaling_summary([baseline, treatment])
    assert len({r["series_id"] for r in summaries}) == 2
    assert next(r for r in summaries if r["gpus"] == 2)["baseline_gpus"] == 2


def test_batch_grid_produces_three_strong_curves_and_one_declared_weak_curve():
    rows = [row(g, b, parameter_scaling_weak_per_gpu_batch_size="4") for b in (4, 8, 16) for g in (1, 2, 4)]
    original = deepcopy(rows)
    summaries = build_scaling_summary(rows)
    strong = [r for r in summaries if r["scaling_type"] == "strong"]
    weak = [r for r in summaries if r["scaling_type"] == "weak"]
    assert len(strong) == 9
    assert len({r["series_id"] for r in strong}) == 3
    assert len(weak) == 3
    assert len({r["series_id"] for r in weak}) == 1
    assert {r["per_gpu_batch_size"] for r in weak} == {"4"}
    assert {r["experiment_ids"] for r in weak} == {"test-1-4", "test-2-8", "test-4-16"}
    assert rows == original


def test_scaling_excludes_diagnostic_and_acceptance_runs():
    summaries = build_scaling_summary(
        [
            row(),
            row(parameter_scaling_role="diagnostic"),
            row(parameter_scaling_role="canary"),
            row(mode="profiling"),
            row(status="failed"),
        ]
    )
    assert len(summaries) == 1
    assert summaries[0]["trials"] == 1


def test_scaling_keeps_full_trial_uncertainty_and_explicit_missing_one_gpu_baseline():
    summaries = build_scaling_summary([row(2, throughput_tokens_sec=str(q)) for q in (774, 780, 9436)])
    assert summaries[0]["trials"] == 3
    assert summaries[0]["throughput_ci95_low"] < summaries[0]["throughput_tokens_sec"]
    assert summaries[0]["throughput_ci95_high"] > summaries[0]["throughput_tokens_sec"]
    assert summaries[0]["baseline_status"] == "relative_to_smallest_observed"
