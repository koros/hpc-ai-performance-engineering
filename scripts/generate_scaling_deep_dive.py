#!/usr/bin/env python3
"""Write the scaling deep-dive configs and execution-matrix waves from their declarative design.

The design lives in ``hpc_ai_perf.config.scaling_deep_dive``. This script owns
``configs/scaling_deep_dive/`` and every ``scaling-dd-*`` wave in
``platforms/execution_matrix.json``; other waves and the file's hand formatting
are left untouched. ``--check`` fails if either is out of date.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hpc_ai_perf.config.scaling_deep_dive import CONFIG_DIR, WAVE_PREFIX, build_config, design, slug, waves

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "platforms" / "execution_matrix.json"


def expected_configs() -> dict[Path, str]:
    entries = design()
    ids = {condition: experiment_id for experiment_id, condition, _ in entries}
    return {
        ROOT / CONFIG_DIR / f"{experiment_id}_{slug(condition)}.json": json.dumps(
            build_config(experiment_id, condition, used_by, ids), indent=2
        )
        + "\n"
        for experiment_id, condition, used_by in entries
    }


def render_wave(wave: dict) -> str:
    ids = ", ".join(f'"{i}"' for i in wave["experiment_ids"])
    return (
        "    {\n"
        f'      "platform": "{wave["platform"]}",\n'
        f'      "wave": "{wave["wave"]}",\n'
        f'      "name": {json.dumps(wave["name"])},\n'
        f'      "experiment_ids": [{ids}],\n'
        f'      "required_trials": {wave["required_trials"]},\n'
        f'      "comparability": "{wave["comparability"]}"\n'
        "    }"
    )


def expected_matrix(text: str) -> str:
    """Replace the generated wave blocks, keeping every other byte of the matrix."""
    block = re.compile(
        r',?\n    \{\n      "platform": "[a-z0-9]+",\n      "wave": "' + re.escape(WAVE_PREFIX) + r"[^}]*?\n    \}"
    )
    stripped = block.sub("", text)
    marker = '\n  ],\n  "deferred"'
    if stripped.count(marker) != 1:
        raise ValueError("Could not locate the end of the waves list in the execution matrix")
    generated = "".join(",\n" + render_wave(wave) for wave in waves(design()))
    updated = stripped.replace(marker, generated + marker)
    data = json.loads(updated)
    ours = [w for w in data["waves"] if w["wave"].startswith(WAVE_PREFIX)]
    if ours != waves(design()):
        raise ValueError("Generated waves did not round-trip through the execution matrix")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if configs or waves differ from the design")
    args = parser.parse_args()
    configs = expected_configs()
    config_dir = ROOT / CONFIG_DIR
    stale = sorted(set(config_dir.glob("E*.json")) - set(configs)) if config_dir.exists() else []
    matrix_text = MATRIX.read_text(encoding="utf-8")
    matrix = expected_matrix(matrix_text)
    if args.check:
        problems = [str(p.relative_to(ROOT)) for p, text in configs.items() if not p.exists() or p.read_text() != text]
        problems += [f"stale {p.relative_to(ROOT)}" for p in stale]
        if matrix != matrix_text:
            problems.append(str(MATRIX.relative_to(ROOT)))
        if problems:
            raise SystemExit("Scaling deep dive is out of date: " + ", ".join(problems[:10]))
        print(f"Scaling deep dive matches its design: {len(configs)} configs.")
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    for path in stale:
        path.unlink()
    for path, text in configs.items():
        if not path.exists() or path.read_text() != text:
            path.write_text(text, encoding="utf-8")
    MATRIX.write_text(matrix, encoding="utf-8")
    print(f"Wrote {len(configs)} scaling deep-dive configs and {len(waves(design()))} waves.")


if __name__ == "__main__":
    main()
