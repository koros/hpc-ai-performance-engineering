#!/usr/bin/env python3
"""Add the verified TinyLlama strategy sweep to the static dashboard snapshot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "dashboard" / "experiments-data.json"
DETAILS = ROOT / "dashboard" / "experiment-details.json"
CATALOG_PAGE = ROOT / "dashboard" / "index.html"
FINDINGS_PAGE = ROOT / "dashboard" / "analysis.html"
TRACKER = ROOT / "results" / "processed" / "experiment_tracker.csv"
TRIALS = ROOT / "results" / "processed" / "experiments.csv"
IDS = tuple(f"E{number}" for number in range(277, 287))
GROUPS = (("h100", IDS[:8], "TinyLlama-1.1B · H100"), ("l40s", IDS[8:], "TinyLlama-1.1B · L40S"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field)]
    if not values:
        return None
    if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field}: incomplete or non-finite values")
    return round(sum(values) / len(values), 4)


def totals(ids: list[str], experiments: dict) -> dict[str, int]:
    counts = {"completed": 0, "configured": 0, "deferred": 0, "total": len(ids)}
    for experiment_id in ids:
        counts[experiments[experiment_id]["status"].lower()] += 1
    return counts


def replace_embedded(page: str, variable: str, payload: str) -> str:
    lines = page.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.lstrip().startswith(f"var {variable} = ")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one embedded {variable} line")
    indent = lines[matches[0]][: len(lines[matches[0]]) - len(lines[matches[0]].lstrip())]
    lines[matches[0]] = f"{indent}var {variable} = {payload};\n"
    return "".join(lines)


def update() -> tuple[dict, dict]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    details = json.loads(DETAILS.read_text(encoding="utf-8"))
    tracker = {row["experiment_id"]: row for row in csv_rows(TRACKER)}
    processed = csv_rows(TRIALS)
    experiments = catalog["experiments"]
    phase = next(
        phase
        for theme in catalog["themes"]
        if theme["id"] == "strategy"
        for phase in theme["phases"]
        if phase["id"] == "distributed-training-strategy"
    )

    for platform, ids, label in GROUPS:
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
            entry = {
                "id": experiment_id,
                "status": "Completed",
                "phase": tracker[experiment_id]["phase"],
                "theme_id": "strategy",
                "theme_label": "Distributed Strategy (RQ2)",
                "workload": "tinyllama",
                "workload_label": "TinyLlama-1.1B",
                "mode": "training",
                "strategy": first["strategy"],
                "strategy_detail": meta.get("strategy_detail"),
                "gpus": first["gpus"],
                "precision": first["precision"],
                "priority": tracker[experiment_id]["priority"],
                "gpu_hours_estimate": tracker[experiment_id]["gpu_hours_estimate"],
                "completed_at": tracker[experiment_id]["completed_at"],
                "runtime_seconds": float(tracker[experiment_id]["runtime_seconds"]),
                "notes": meta.get("limitation"),
                "model": first["model"],
                "scaling_type": None,
                "comparability": "exact",
                "framework": None,
                "batch_size": first["batch_size"],
                "global_batch_size": first["global_batch_size"],
                "per_gpu_batch_size": first["per_gpu_batch_size"],
                "quantization_method": None,
                "sequence_length": str(parameters["sequence_length"]),
                "prompt_length": None,
                "concurrent_requests": None,
                "max_new_tokens": None,
                "gradient_accumulation_steps": first.get("parameter_gradient_accumulation_steps") or None,
                "activation_checkpointing": first.get("parameter_activation_checkpointing") or None,
                "pinned_memory": None,
                "num_workers": None,
                "research_questions": meta["research_questions"],
                "platforms": {platform: measured},
                "platform_list": [platform],
            }
            experiments[experiment_id] = entry
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
                "status": "Completed",
                "strategy": entry["strategy"],
                "strategy_detail": entry["strategy_detail"],
                "workload": entry["workload"],
            }

        key = f"tinyllama_{platform}"
        cluster = next((item for item in phase["clusters"] if item["key"] == key), None)
        if cluster is None:
            cluster = {"key": key, "label": label}
            phase["clusters"].append(cluster)
        cluster.update(
            {
                "totals": totals(list(ids), experiments),
                "constant": {
                    "workload_label": "TinyLlama-1.1B",
                    "model": experiments[ids[0]]["model"],
                    "mode": "training",
                    "precision": "bf16",
                    "platform_list": [platform],
                    "batch_size": experiments[ids[0]]["batch_size"],
                    "per_gpu_batch_size": experiments[ids[0]]["per_gpu_batch_size"],
                    "sequence_length": "1024",
                    "comparability": "exact",
                    **({"gpus": "8", "global_batch_size": "16"} if platform == "l40s" else {}),
                },
                "varied": {
                    "strategy": sorted({experiments[eid]["strategy"] for eid in ids}),
                    "strategy_detail": sorted({experiments[eid]["strategy_detail"] for eid in ids}),
                    **(
                        {
                            "gpus": ["2", "4"],
                            "global_batch_size": ["16", "32"],
                        }
                        if platform == "h100"
                        else {}
                    ),
                },
                "experiment_ids": list(ids),
            }
        )

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
    catalog["meta"]["generated_at"] = "2026-09-30"
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
            raise SystemExit("Stale TinyLlama strategy dashboard data: " + ", ".join(stale))
        print("TinyLlama strategy dashboard data matches processed evidence.")
        return
    for path, contents in expected.items():
        path.write_text(contents, encoding="utf-8")
    print("Updated E277-E286 from three completed trials per platform condition.")


if __name__ == "__main__":
    main()
