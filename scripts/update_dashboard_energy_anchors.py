#!/usr/bin/env python3
"""Synchronize the static catalog's L40S hardware-energy anchors with verified evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from verify_hardware_energy_revalidation import EXPECTED_REFERENCES, REQUIRED_TRIALS, verify_results

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "dashboard" / "experiments-data.json"
PAGE = ROOT / "dashboard" / "index.html"
TRACKER = ROOT / "results" / "processed" / "experiment_tracker.csv"
TRIALS = ROOT / "results" / "processed" / "experiments.csv"
RAW = ROOT / "results" / "raw" / "l40s"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(records: list[dict[str, str]], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key)]
    if not values:
        return None
    if len(values) != REQUIRED_TRIALS or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{key}: expected {REQUIRED_TRIALS} finite measurements")
    return round(sum(values) / len(values), 4)


def _totals(ids: list[str], experiments: dict) -> dict[str, int]:
    counts = {"completed": 0, "configured": 0, "deferred": 0, "total": len(ids)}
    for experiment_id in ids:
        counts[experiments[experiment_id]["status"].lower()] += 1
    return counts


def updated_catalog() -> dict:
    issues, accepted = verify_results(RAW)
    if issues:
        raise ValueError("Hardware energy evidence failed verification:\n- " + "\n- ".join(issues))

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    experiments = catalog["experiments"]
    tracker = {row["experiment_id"]: row for row in _csv_rows(TRACKER)}
    trials = _csv_rows(TRIALS)

    for experiment_id, reference_id in EXPECTED_REFERENCES.items():
        records = [row for row in trials if row["experiment_id"] == experiment_id and row["platform_id"] == "l40s"]
        raw_names = {path.name for path in (RAW / experiment_id).glob("*.json")}
        processed_names = {Path(row["source_path"]).name for row in records}
        if len(records) != REQUIRED_TRIALS or raw_names != processed_names:
            raise ValueError(f"{experiment_id}: processed and verified raw trial sets do not match")
        if tracker[experiment_id]["status"] != "Completed" or any(row["status"] != "completed" for row in records):
            raise ValueError(f"{experiment_id}: tracker or trial status is not completed")
        if any(row["energy_reference_experiment_id"] != reference_id for row in records):
            raise ValueError(f"{experiment_id}: energy reference differs from {reference_id}")

        entry = experiments[experiment_id]
        reference = experiments[reference_id]
        entry["status"] = "Completed"
        entry["completed_at"] = tracker[experiment_id]["completed_at"]
        entry["runtime_seconds"] = float(tracker[experiment_id]["runtime_seconds"])
        for field in (
            "model",
            "framework",
            "batch_size",
            "global_batch_size",
            "per_gpu_batch_size",
            "sequence_length",
            "prompt_length",
            "comparability",
        ):
            value = records[0].get(field)
            entry[field] = value if value and value != "None" else reference.get(field)
        entry["research_questions"] = reference["research_questions"]
        entry["energy_reference_experiment_id"] = reference_id

        metrics = {"n": REQUIRED_TRIALS}
        for key in (
            "throughput_tokens_sec",
            "throughput_samples_sec",
            "memory_used_gb",
            "latency_p50_ms",
            "latency_p95_ms",
            "runtime_seconds",
        ):
            value = _mean(records, key)
            if value is not None:
                metrics[key] = value
        measured_energy = accepted[experiment_id]["avg_energy_joules"]
        if not math.isclose(_mean(records, "energy_joules"), measured_energy, abs_tol=5e-5):
            raise ValueError(f"{experiment_id}: processed energy differs from verified energy")
        metrics["energy_joules"] = round(measured_energy, 4)
        metrics["power_draw_watts"] = round(accepted[experiment_id]["avg_power_watts"], 4)
        metrics["gpu_model"] = records[0]["gpu_model"]
        entry["platforms"] = {"l40s": metrics}
        entry["platform_list"] = ["l40s"]

    for theme in catalog["themes"]:
        phase_ids: list[str] = []
        for phase in theme["phases"]:
            ids: list[str] = []
            for cluster in phase["clusters"]:
                cluster["totals"].update(_totals(cluster["experiment_ids"], experiments))
                ids.extend(cluster["experiment_ids"])
            phase["totals"].update(_totals(ids, experiments))
            phase_ids.extend(ids)
        theme["totals"].update(_totals(phase_ids, experiments))
    catalog["meta"].update(_totals(list(experiments), experiments))
    catalog["meta"]["generated_at"] = "2026-09-30"
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the static catalog is stale")
    args = parser.parse_args()

    payload = json.dumps(updated_catalog(), ensure_ascii=True, separators=(",", ":"))
    catalog_text = payload + "\n"
    lines = PAGE.read_text(encoding="utf-8").splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith("var DATA = ")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one embedded dashboard DATA line")
    lines[matches[0]] = "var DATA = " + payload + ";\n"
    page_text = "".join(lines)

    if args.check:
        if CATALOG.read_text(encoding="utf-8") != catalog_text or PAGE.read_text(encoding="utf-8") != page_text:
            raise SystemExit("Dashboard energy anchors are stale; run scripts/update_dashboard_energy_anchors.py")
        print("Dashboard energy anchors match verified evidence.")
        return

    CATALOG.write_text(catalog_text, encoding="utf-8")
    PAGE.write_text(page_text, encoding="utf-8")
    print("Updated E235-E238 in the catalog from verified L40S energy evidence.")


if __name__ == "__main__":
    main()
