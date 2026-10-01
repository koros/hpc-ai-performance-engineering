#!/usr/bin/env python3
"""Sync study knobs from experiment configs into the dashboard catalog snapshot.

The catalog tables build their columns from each group's ``varied`` parameters. Some knobs were
never copied into the catalog (``prefetch_factor``), are absent from the group's ``varied`` list
(``pinned_memory``, ``num_workers``, gradient accumulation, activation checkpointing), or were
blanked for deferred conditions (``framework``). This script restores them from ``configs/`` so a
reader can see what each condition changed. Measured results are not touched.
"""

from __future__ import annotations

import argparse
import json

from update_dashboard_tinyllama_strategy import CATALOG, CATALOG_PAGE, ROOT, replace_embedded

# Knobs whose distinct non-null values within a group mark that group as varying them.
VARIED_KNOBS = (
    "framework",
    "pinned_memory",
    "num_workers",
    "prefetch_factor",
    "gradient_accumulation_steps",
    "activation_checkpointing",
)


def load_configs() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for path in sorted((ROOT / "configs").rglob("E*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        experiment_id = config.get("experiment_id")
        if experiment_id and experiment_id not in configs:
            configs[experiment_id] = config
    return configs


def update() -> dict:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    configs = load_configs()
    for experiment_id, experiment in catalog["experiments"].items():
        config = configs.get(experiment_id)
        if config is None:
            continue
        parameters = config.get("workload", {}).get("parameters", {})
        if "prefetch_factor" in parameters:
            experiment["prefetch_factor"] = str(parameters["prefetch_factor"])
        if experiment.get("framework") is None and parameters.get("framework"):
            experiment["framework"] = parameters["framework"]
        objective = config.get("metadata", {}).get("objective")
        if objective:
            experiment["objective"] = objective

    for theme in catalog["themes"]:
        for phase in theme["phases"]:
            for cluster in phase["clusters"]:
                members = [catalog["experiments"][eid] for eid in cluster["experiment_ids"]]
                for knob in VARIED_KNOBS:
                    values = sorted({m[knob] for m in members if m.get(knob) is not None})
                    if len(values) > 1:
                        cluster["varied"][knob] = values
                        cluster["constant"].pop(knob, None)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the static data is stale")
    args = parser.parse_args()
    payload = json.dumps(update(), ensure_ascii=True, separators=(",", ":"))
    expected = {
        CATALOG: payload + "\n",
        CATALOG_PAGE: replace_embedded(CATALOG_PAGE.read_text(encoding="utf-8"), "DATA", payload),
    }
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, text in expected.items() if p.read_text(encoding="utf-8") != text]
        if stale:
            raise SystemExit("Stale catalog study context: " + ", ".join(stale))
        print("Catalog study context matches experiment configs.")
        return
    for path, contents in expected.items():
        path.write_text(contents, encoding="utf-8")
    print("Synced study knobs and objectives from configs into the catalog.")


if __name__ == "__main__":
    main()
