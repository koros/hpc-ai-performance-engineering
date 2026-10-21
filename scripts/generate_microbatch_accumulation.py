"""Generate the fixed-effective-batch follow-up without changing historical configs."""

import argparse
import json
from pathlib import Path

from hpc_ai_perf.config.microbatch_accumulation import CANARY_WAVE, CONFIG_ROOT, PERFORMANCE_WAVE, design


def expected_files():
    rows = design()
    files = {
        Path(CONFIG_ROOT) / f"{eid}_{data['experiment']}.json": json.dumps(data, indent=2) + "\n"
        for eid, _, data in rows
    }
    path = Path("platforms/execution_matrix.json")
    matrix = json.loads(path.read_text())
    waves = {CANARY_WAVE, PERFORMANCE_WAVE}
    matrix["waves"] = [w for w in matrix["waves"] if w["wave"] not in waves]
    matrix["waves"] += [
        dict(
            platform="l40s",
            wave=wave,
            name=f"L40S {wave}",
            experiment_ids=[eid for eid, w, _ in rows if w == wave],
            required_trials=1 if wave == CANARY_WAVE else 3,
            comparability="exact",
        )
        for wave in (CANARY_WAVE, PERFORMANCE_WAVE)
    ]
    # Preserve hand formatting of every unrelated wave.
    import re

    try:
        from scripts.generate_scaling_deep_dive import render_wave
    except ModuleNotFoundError:
        from generate_scaling_deep_dive import render_wave
    text = path.read_text()
    for wave in waves:
        text = re.sub(r',?\n    \{\n      "platform": "l40s",\n      "wave": "' + wave + r'"[^}]*?\n    \}', "", text)
    marker = re.search(r'    \{\n      "platform": "[a-z0-9]+",\n      "wave": "scaling-batch-canary', text)
    if marker is None:
        raise ValueError("Cannot locate the scaling-batch wave section")
    generated = ",\n".join(render_wave(w) for w in matrix["waves"][-2:])
    text = text[: marker.start()] + generated + ",\n" + text[marker.start() :]
    parsed = json.loads(text)
    assert [w for w in parsed["waves"] if w["wave"] in waves] == matrix["waves"][-2:]
    files[path] = text
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = expected_files()
    if args.check:
        stale = [str(p) for p, text in files.items() if not p.exists() or p.read_text() != text]
        if stale:
            raise SystemExit("Microbatch follow-up drift: " + ", ".join(stale))
    else:
        for p, text in files.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    print("Microbatch follow-up: 6 acceptance conditions; 12 performance conditions x 3 trials")


if __name__ == "__main__":
    main()
