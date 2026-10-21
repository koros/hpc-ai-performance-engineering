import csv
import json
from pathlib import Path

import pytest

from hpc_ai_perf.analysis.plot_data import (
    METRICS,
    clean,
    export_plot_data,
    normalize,
    sample_rows,
    select_cohorts,
    summarize,
    write_table,
)


def trial(platform="l40s", run="r1", throughput=10, cohort="a", when="2026-10-21T10:00:00Z", status="completed"):
    row = dict(
        platform=platform,
        experiment_id="E771",
        run_id=run,
        submission_cohort_id=cohort,
        completed_at=when,
        started_at=when,
        status=status,
        error=None,
        git_dirty=False,
        role="performance",
    )
    row.update({metric: None for metric in METRICS})
    row["throughput_tokens_sec"] = throughput
    return row


def test_latest_failed_cohort_is_not_hidden_by_older_success():
    old = trial(cohort="old", when="2026-10-20T10:00:00Z")
    failed = trial(cohort="new", status="failed")
    select_cohorts([old, failed])
    assert not old["analysis_eligible"] and not failed["analysis_eligible"]
    assert failed["selected_current_cohort"] and not old["selected_current_cohort"]
    assert summarize([old, failed]) == []


def test_condition_summaries_use_independent_trials_without_platform_pooling():
    rows = [trial(run=f"r{i}", throughput=v) for i, v in enumerate((10, 20, 30))]
    rows += [trial(platform="h100", run="h1", throughput=100)]
    select_cohorts(rows)
    summaries = {r["platform"]: r for r in summarize(rows)}
    assert summaries["l40s"]["throughput_tokens_sec_mean"] == 20
    assert summaries["l40s"]["throughput_tokens_sec_n"] == 3
    assert summaries["l40s"]["throughput_tokens_sec_sd"] == 10
    assert summaries["l40s"]["throughput_tokens_sec_ci95_low"] < 20
    assert summaries["h100"]["throughput_tokens_sec_mean"] == 100
    assert summaries["h100"]["throughput_tokens_sec_ci95_low"] is None
    assert summaries["h100"]["throughput_tokens_sec_ci95_high"] is None
    assert summaries["h100"]["joules_per_output_token_mean"] is None


def test_canaries_and_dirty_source_do_not_enter_performance_summaries():
    canary, dirty = trial(run="canary"), trial(run="dirty")
    canary["role"] = "canary"
    dirty["git_dirty"] = True
    select_cohorts([canary, dirty])
    assert summarize([canary, dirty]) == []


def test_normalization_uses_measured_energy_and_retains_follow_up_note(tmp_path):
    path = tmp_path / "l40s/E771/result.json"
    path.parent.mkdir(parents=True)
    record = dict(
        run_id="r1",
        experiment_id="E771",
        mode="inference",
        platform_id="l40s",
        metrics={"energy_joules_measured_region": 100, "energy_joules_estimate": 9999, "generated_tokens": 50},
        config={
            "workload": {"parameters": {"concurrent_requests": 2}},
            "metadata": {"study_design_note": "follow-up motivated by observed results"},
        },
    )
    path.write_text(json.dumps(record))
    row = normalize(record, path, tmp_path)
    assert row["joules_per_output_token"] == 2
    assert row["concurrency"] == 2
    assert row["energy_scope"] == "measured_region"
    assert row["study_design_note"] == "follow-up motivated by observed results"
    assert row["source_record"] == "l40s/E771/result.json"
    assert len(row["source_sha256"]) == 64


def test_csv_json_have_matching_flat_values_and_finite_numbers(tmp_path):
    schema = write_table(tmp_path, "demo/table", [{"value": 1.5, "missing": None, "flag": True, "bad": float("nan")}])
    rows = json.loads((tmp_path / schema["json"]).read_text())
    assert rows == [{"value": 1.5, "missing": None, "flag": True, "bad": None}]
    assert isinstance(rows[0]["value"], float)
    with (tmp_path / schema["csv"]).open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["value"]) == rows[0]["value"] and row["flag"] == "true"
    assert row["missing"] == row["bad"] == ""
    assert schema["schema"]["value"]["types"] == ["number"]
    assert clean(float("inf")) is None


def test_samples_remain_long_form_and_keep_trial_identity():
    source = {"samples": {"time_to_first_token_seconds": [0.01, 0.02], "decode_time_seconds": [0.1]}}
    selected = [
        {
            "experiment_id": "E771",
            "mode": "inference",
            "source_record": "source",
            "run_id": "r1",
            "platform": "l40s",
            "model": "qwen",
            "serving_layout": "tp2_dp4",
            "concurrency": 2,
            "runtime_cohort_id": "runtime",
        }
    ]
    rows = sample_rows(selected, {"source": source})
    assert len(rows) == 3 and {r["run_id"] for r in rows} == {"r1"}
    assert {r["unit"] for r in rows} == {"seconds"}
    assert [r["sample_index"] for r in rows] == [0, 1, 0]


def test_export_cannot_write_into_raw_evidence(tmp_path):
    with pytest.raises(ValueError, match="separate from raw"):
        export_plot_data(tmp_path / "raw", tmp_path / "raw/exports", Path("platforms/execution_matrix.json"))
