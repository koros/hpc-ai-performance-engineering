"""Guard the current study aggregates and their offline publication."""

import json
import subprocess
import sys
from pathlib import Path

from scripts.build_dashboard_site import DASHBOARD_FILES
from scripts.build_dashboard_studies import load_sources, ratio_ci, verdict_of


def test_study_headlines_count_each_condition_platform_trial_once():
    data = json.loads(Path("dashboard/studies-data.json").read_text())
    _, _, _, coverage, _, _ = load_sources()
    ids = {eid for study in data["studies"] for eid in study["ids"]}
    assert data["meta"]["study_conditions"] == len(ids)
    assert data["meta"]["study_completed_trials"] == sum(
        int(row["completed_trials"] or 0) for (eid, _), row in coverage.items() if eid in ids
    )
    assert len(ids) < sum(len(study["ids"]) for study in data["studies"])
    assert {"studies.html", "studies-data.json"} <= set(DASHBOARD_FILES)


def test_studies_are_reproducible_and_embedded_for_offline_reading():
    subprocess.run([sys.executable, "scripts/build_dashboard_studies.py", "--check"], check=True)
    page = Path("dashboard/studies.html").read_text()
    embedded = next(line for line in page.splitlines() if line.startswith("var STUDIES = "))
    assert json.loads(embedded.removeprefix("var STUDIES = ").removesuffix(";")) == json.loads(
        Path("dashboard/studies-data.json").read_text()
    )


def test_uncertain_effects_do_not_become_positive_findings():
    assert verdict_of(5, -1, 11) == "inconclusive"
    assert verdict_of(5, 4, 6) == "helps"
    assert verdict_of(-5, -6, -4) == "hurts"
    assert verdict_of(0, -1, 1) == "no_material_effect"
    assert ratio_ci(10, None, 3, 5, None, 3) == (2, None, None)


def test_student_t_quantiles_preserve_fractional_degrees_of_freedom():
    import pytest
    from scipy.stats import t
    from scripts.build_dashboard_studies import tcrit

    from hpc_ai_perf.analysis.scaling_deep_dive import _t95

    for dof in (1, 2, 2.1, 3.5, 7, 9, 31, 100):
        expected = t.ppf(0.975, dof)
        assert _t95(dof) == pytest.approx(expected, rel=1e-12)
        assert tcrit(dof) == pytest.approx(expected, rel=1e-12)
    assert _t95(2.1) > 4.0  # Rounding up to df=3 previously returned 3.182.
    assert _t95(31) > 1.96  # Finite df must not fall back to the normal quantile.


def test_distilbert_grad_op_effect_is_inconclusive_in_published_study():
    import csv

    data = json.loads(Path("dashboard/studies-data.json").read_text())
    study = next(s for s in data["studies"] if s["id"] == "s5-stacks")
    rows = [r for table in study["tables"] for r in table["rows"]]
    row = next(r for r in rows if r.get("treatment_id") == "E589" and r.get("platform") == "L40S")
    assert row["verdict"] == "inconclusive"
    assert row["lo"] < 0 < row["hi"]
    with Path("results/plot_data/training_parameter_effects/training_stack_effects.csv").open() as handle:
        effect = next(
            r for r in csv.DictReader(handle) if r["treatment_experiment_id"] == "E589" and r["platform_id"] == "l40s"
        )
    assert effect["verdict"] == "inconclusive"
    assert float(effect["change_ci95_low"]) < 0 < float(effect["change_ci95_high"])


def test_every_exported_effect_matches_current_trials_and_welch_interval():
    import csv
    import math
    import statistics
    from collections import defaultdict

    import pytest
    from scipy.stats import t

    trials = defaultdict(list)
    with Path("results/plot_data/trial_level/current_performance_trials.csv").open() as handle:
        for row in csv.DictReader(handle):
            trials[row["platform"], row["experiment_id"]].append(float(row["throughput_tokens_sec"]))
    checked = 0
    identities = set()
    for path in Path("results/plot_data/training_parameter_effects").glob("*_effects.csv"):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                control = trials[row["platform_id"], row["control_experiment_id"]]
                treatment = trials[row["platform_id"], row["treatment_experiment_id"]]
                mean = statistics.fmean(control)
                change = 100 * (statistics.fmean(treatment) / mean - 1)
                assert float(row["throughput_change_pct"]) == pytest.approx(change, abs=1e-6)
                checked += 1
                identities.add(
                    (row["lever"], row["platform_id"], row["control_experiment_id"], row["treatment_experiment_id"])
                )
                if min(len(control), len(treatment)) < 2:
                    assert row["change_ci95_low"] == row["change_ci95_high"] == ""
                    assert row["verdict"] == "inconclusive"
                    continue
                vc = statistics.variance(control) / len(control)
                vt = statistics.variance(treatment) / len(treatment)
                if vc + vt:
                    df = (vc + vt) ** 2 / (vc**2 / (len(control) - 1) + vt**2 / (len(treatment) - 1))
                    margin = t.ppf(0.975, df) * math.sqrt(vc + vt) / mean * 100
                else:
                    margin = 0
                assert float(row["change_ci95_low"]) == pytest.approx(change - margin, abs=1e-6)
                assert float(row["change_ci95_high"]) == pytest.approx(change + margin, abs=1e-6)
    assert checked == 540  # Combined table plus its per-lever exports.
    assert len(identities) == 270
