#!/usr/bin/env python3
"""Add the verified TinyLlama weak-scaling wave to the static dashboard snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from update_dashboard_tinyllama_strategy import (
    CATALOG,
    CATALOG_PAGE,
    DETAILS,
    FINDINGS_PAGE,
    ROOT,
    TRACKER,
    TRIALS,
    csv_rows,
    mean,
    replace_embedded,
    totals,
)

GROUPS = (("h100", ("E287", "E288", "E289")), ("l40s", ("E290", "E291", "E292", "E293")))
GENERATED_AT = "2026-10-01"


def update() -> tuple[dict, dict]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    details = json.loads(DETAILS.read_text(encoding="utf-8"))
    tracker = {row["experiment_id"]: row for row in csv_rows(TRACKER)}
    processed = csv_rows(TRIALS)
    experiments = catalog["experiments"]

    for platform, ids in GROUPS:
        for experiment_id in ids:
            rows = [
                row for row in processed if row["experiment_id"] == experiment_id and row["platform_id"] == platform
            ]
            raw_paths = sorted((ROOT / "results" / "raw" / platform / experiment_id).glob("*.json"))
            if (
                len(rows) != 3
                or len(raw_paths) != 3
                or {Path(row["source_path"]).name for row in rows} != {path.name for path in raw_paths}
            ):
                raise ValueError(f"{experiment_id}@{platform}: expected three matching raw and processed trials")
            raw = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
            meta = raw[0]["config"]["metadata"]
            if (
                tracker[experiment_id]["status"] != "Completed"
                or any(row["status"] != "completed" or row["comparability"] != "exact" for row in rows)
                or any(
                    record["status"] != "completed" or record["config"]["metadata"]["comparability"] != "exact"
                    for record in raw
                )
                or any(record["config"]["metadata"]["execution_platform"] != platform for record in raw)
            ):
                raise ValueError(f"{experiment_id}@{platform}: incomplete or non-comparable evidence")

            first = rows[0]
            measured = {"n": len(rows)}
            for field in (
                "throughput_tokens_sec",
                "throughput_samples_sec",
                "memory_used_gb",
                "energy_joules",
                "runtime_seconds",
            ):
                value = mean(rows, field)
                if value is not None:
                    measured[field] = value
            measured["gpu_model"] = first["gpu_model"]
            if not measured.get("throughput_tokens_sec", 0) > 0 or not measured.get("memory_used_gb", 0) > 0:
                raise ValueError(f"{experiment_id}: missing training throughput or memory")

            parameters = raw[0]["config"]["workload"]["parameters"]
            entry = experiments[experiment_id]
            entry.update(
                {
                    "status": "Completed",
                    "completed_at": tracker[experiment_id]["completed_at"],
                    "runtime_seconds": measured["runtime_seconds"],
                    "model": first["model"],
                    "scaling_type": "weak",
                    "comparability": "exact",
                    "batch_size": first["batch_size"],
                    "global_batch_size": first["global_batch_size"],
                    "per_gpu_batch_size": first["per_gpu_batch_size"],
                    "sequence_length": str(parameters["sequence_length"]),
                    "research_questions": meta["research_questions"],
                    "platforms": {platform: measured},
                    "platform_list": [platform],
                }
            )
            details[experiment_id] = {
                "batch_size": entry["batch_size"],
                "comparability": entry["comparability"],
                "completed_at": entry["completed_at"][:10],
                "global_batch_size": entry["global_batch_size"],
                "gpus": entry["gpus"],
                "mode": entry["mode"],
                "model": entry["model"],
                "per_gpu_batch_size": entry["per_gpu_batch_size"],
                "phase": entry["phase"],
                "platforms": {
                    f"lovelace_{platform}": {
                        "n": 3,
                        "gpu_model": measured["gpu_model"],
                        "throughput_tokens_sec": measured["throughput_tokens_sec"],
                        "memory_used_gb": measured["memory_used_gb"],
                        "runtime_seconds": measured["runtime_seconds"],
                        **({"energy_j": measured["energy_joules"]} if "energy_joules" in measured else {}),
                    }
                },
                "precision": entry["precision"],
                "scaling_type": "weak",
                "status": "Completed",
                "strategy": entry["strategy"],
                "workload": entry["workload"],
            }

    for theme in catalog["themes"]:
        theme_ids = []
        for current_phase in theme["phases"]:
            phase_ids = []
            for cluster in current_phase["clusters"]:
                cluster["totals"] = totals(cluster["experiment_ids"], experiments)
                phase_ids.extend(cluster["experiment_ids"])
            current_phase["totals"] = totals(phase_ids, experiments)
            theme_ids.extend(phase_ids)
        theme["totals"] = totals(theme_ids, experiments)
    catalog["meta"].update(totals(list(experiments), experiments))
    catalog["meta"]["generated_at"] = GENERATED_AT
    return catalog, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the static data is stale")
    args = parser.parse_args()
    catalog, details = update()
    catalog_payload = json.dumps(catalog, ensure_ascii=True, separators=(",", ":"))
    details_payload = json.dumps(details, ensure_ascii=True, separators=(",", ":"))
    expected = {
        CATALOG: catalog_payload + "\n",
        DETAILS: details_payload + "\n",
        CATALOG_PAGE: replace_embedded(CATALOG_PAGE.read_text(encoding="utf-8"), "DATA", catalog_payload),
        FINDINGS_PAGE: replace_embedded(
            FINDINGS_PAGE.read_text(encoding="utf-8"), "EXPERIMENT_DETAILS", details_payload
        ),
    }
    if args.check:
        stale = [
            str(path.relative_to(ROOT))
            for path, contents in expected.items()
            if path.read_text(encoding="utf-8") != contents
        ]
        if stale:
            raise SystemExit("Stale TinyLlama weak-scaling dashboard data: " + ", ".join(stale))
        print("TinyLlama weak-scaling dashboard data matches processed evidence.")
        return
    for path, contents in expected.items():
        path.write_text(contents, encoding="utf-8")
    print("Updated E287-E293 from three completed trials per condition.")


if __name__ == "__main__":
    main()
