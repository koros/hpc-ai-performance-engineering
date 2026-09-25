from __future__ import annotations

import json
from copy import deepcopy

import pytest

from hpc_ai_perf.analysis.lowbit import build_lowbit_summary
from hpc_ai_perf.analysis.summaries import build_analysis_tables, build_precision_summary
from hpc_ai_perf.config.lowbit import digest
from hpc_ai_perf.profiling.system import _NvidiaSmiSample, _summarize_nvidia_smi_window
from hpc_ai_perf.results import ExperimentResult
from hpc_ai_perf.results.aggregate import flatten_result_record
from hpc_ai_perf.validation.lowbit import resolved_lowbit_config


def trials(experiment="E239", platform="l40s", throughput=100, quality=10, memory=15):
    config = resolved_lowbit_config(experiment, platform)
    rows = []
    for repetition in (1, 2, 3):
        result = ExperimentResult.from_config(
            config,
            status="completed",
            started_at="2026-09-25T12:00:00Z",
            completed_at="2026-09-25T12:02:00Z",
            repetition_number=repetition,
            runtime_seconds=120,
            metrics={
                "throughput_tokens_sec": throughput,
                "throughput_unit": "generated_tokens_per_second",
                "study_runtime_sha256": "runtime",
                "runtime_packages": '{"vllm":"0.10.2"}',
                "performance_tokens_sha256": "prompts",
                "quality_tokens_sha256": "quality",
                "base_model_revision": "base",
                "vllm_execution_mode": "v0_eager",
                "quantization_method": config.workload.parameters["quantization_method"],
                "batch_size": config.workload.parameters["batch_size"],
                "quality_perplexity": quality,
                "latency_p95_ms": 100,
                "time_to_first_token_p95_ms": 20,
                "memory_model_weights_gb": memory,
                "memory_worker_peak_allocated_gb": memory + 2,
                "generated_tokens": 1000,
                "energy_joules_measured_region": 200,
                "energy_scope": "measured_region",
                "energy_measurement_status": "sufficient",
                "nvidia_smi.gpu_mapping_method": "cuda_uuid",
                "nvidia_smi.measurement_interior_samples": 100,
                "nvidia_smi.measurement_max_gap_seconds": 0.3,
            },
            environment={
                "config_sha256": digest(config.to_dict()),
                "git_dirty": False,
                "observed_gpu_model": f"NVIDIA {platform.upper()}",
            },
        )
        flat = flatten_result_record(result.to_dict(), f"{experiment}-{repetition}.json")
        # Match the CSV boundary used by the production analysis pipeline.
        rows.append(
            {
                key: json.dumps(value) if isinstance(value, list | dict) else str(value) if value is not None else ""
                for key, value in flat.items()
            }
        )
    return rows


def test_analysis_compares_matched_controls_with_quality_and_energy_per_token():
    rows = trials() + trials("E241", throughput=120, quality=10.5, memory=5)
    output = {r["experiment_id"]: r for r in build_lowbit_summary(rows)}
    treatment = output["E241"]
    assert treatment["evidence_status"] == "complete"
    assert treatment["throughput_change_pct"] == pytest.approx(20)
    assert treatment["model_memory_saving_pct"] == pytest.approx(100 * 2 / 3)
    assert treatment["perplexity_change_pct"] == pytest.approx(5)
    assert treatment["quality_status"] == "within_tolerance"
    assert treatment["mean_joules_per_output_token"] == pytest.approx(0.2)
    assert treatment["energy_trials"] == 3
    assert output["E242"]["evidence_status"] == "not_run"
    assert build_precision_summary(rows) == []
    assert "lowbit_summary" in build_analysis_tables(rows)


def test_platform_and_load_are_never_used_as_each_others_control():
    rows = trials("E239", "h100") + trials("E241", "l40s") + trials("E242", "h100")
    treatments = [r for r in build_lowbit_summary(rows) if r["completed_trials"] and r["method"] == "awq"]
    assert {r["evidence_status"] for r in treatments} == {"control_incomplete"}
    assert all(r["throughput_change_pct"] is None for r in treatments)


def test_analysis_rejects_protocol_drift_and_duplicate_repetitions():
    control = trials()
    treatment = trials("E241")
    for row in treatment:
        row["metric_quality_tokens_sha256"] = "different dataset"
    output = next(r for r in build_lowbit_summary(control + treatment) if r["experiment_id"] == "E241")
    assert output["evidence_status"] == "protocol_mismatch"
    assert output["throughput_change_pct"] is None
    duplicate = trials("E241")
    duplicate.append(deepcopy(duplicate[0]))
    output = next(r for r in build_lowbit_summary(control + duplicate) if r["experiment_id"] == "E241")
    assert output["evidence_status"] == "ambiguous_repetitions"


def test_quality_failure_is_a_scientific_result_not_a_faster_winner():
    output = next(
        r
        for r in build_lowbit_summary(trials() + trials("E241", throughput=200, quality=12))
        if r["experiment_id"] == "E241"
    )
    assert output["evidence_status"] == "complete"
    assert output["quality_status"] == "outside_tolerance"
    assert "do not recommend on speed alone" in output["interpretation"]


def test_sparse_energy_is_unavailable_without_discarding_valid_throughput():
    treatment = trials("E241")
    for row in treatment:
        row["metric_nvidia_smi_measurement_max_gap_seconds"] = 15
    output = next(r for r in build_lowbit_summary(trials() + treatment) if r["experiment_id"] == "E241")
    assert output["evidence_status"] == "complete"
    assert output["energy_trials"] == 0
    assert output["mean_joules_per_output_token"] is None
    assert output["energy_per_token_change_pct"] is None


def test_power_coverage_counts_interior_samples_not_only_outside_anchors():
    samples = [
        _NvidiaSmiSample(
            timestamp=t, utilization=10, memory_mib=100, power_watts=100, temperature_c=40, sm_clock_mhz=1000
        )
        for t in (-1, 31)
    ]
    summary = _summarize_nvidia_smi_window(samples, 0, 30, require_two_observations=True)
    assert summary["source_samples"] == 2
    assert summary["interior_samples"] == 0
    assert summary["max_gap_seconds"] == 30
