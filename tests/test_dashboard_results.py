import json
import statistics
from pathlib import Path

from scripts.update_dashboard_results import build_catalog, inference_section, update

from hpc_ai_perf.config import discover_experiment_configs


def test_full_catalog_has_each_config_once_and_no_false_pending_results():
    catalog, details = build_catalog()
    expected_ids = {entry.matrix_id for entry in discover_experiment_configs()}
    assert catalog["meta"]["total"] == len(expected_ids)
    grouped = [
        eid
        for theme in catalog["themes"]
        for phase in theme["phases"]
        for cluster in phase["clusters"]
        for eid in cluster["experiment_ids"]
    ]
    assert len(grouped) == len(set(grouped)) == len(expected_ids)
    assert set(grouped) == expected_ids
    assert set(grouped) == set(catalog["experiments"])
    for eid in ("E765", "E766", "E767", "E768"):
        if not list(Path("results/raw/l40s", eid).glob("*.json")):
            assert catalog["experiments"][eid]["status"] == "Configured"
            assert details[eid]["platforms"] == {}
    assert catalog["experiments"]["E719"]["platforms"]["h100"]["n"] == 1
    assert catalog["experiments"]["E719"]["status"] == "Completed"


def test_current_inference_values_match_raw_per_platform_trial_means():
    catalog, details = build_catalog()
    for eid in ("E722", "E737", "E752", "E758", "E762", "E763", "E764"):
        rows = [json.loads(p.read_text()) for p in Path("results/raw/l40s", eid).glob("*.json")]
        metrics = catalog["experiments"][eid]["platforms"]["l40s"]
        assert metrics["n"] == len(rows) == 3
        assert metrics["throughput_tokens_sec"] == statistics.mean(r["metrics"]["throughput_tokens_sec"] for r in rows)
        assert metrics["joules_per_output_token"] == statistics.mean(
            r["metrics"]["energy_joules_measured_region"] / r["metrics"]["generated_tokens"] for r in rows
        )
        assert details[eid]["platforms"]["lovelace_l40s"]["n"] == 3
    html = inference_section(catalog)
    assert "E764@l40s" in html
    if not list(Path("results/raw/l40s/E767").glob("*.json")):
        assert "E767@l40s" not in html
    assert "E759@l40s" not in html  # canary excluded
    assert "p95 values are means of within-trial percentiles" in html


def test_dashboard_refresh_is_current_and_repeatable():
    expected = update()
    for path, content in expected.items():
        assert path.read_text() == content, f"Refresh {path}"
