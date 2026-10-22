"""Declare a focused L40S S7 acceptance wave without changing historical configs."""

import argparse
import json
import re
from pathlib import Path

from hpc_ai_perf.config.s7_real_text import CONFIG_ROOT, design

WAVE = "s7-real-text-canary"


def expected_matrix():
    path = Path("platforms/execution_matrix.json")
    text = path.read_text()
    text = re.sub(r',?\n    \{\n      "platform": "l40s",\n      "wave": "s7-real-text-canary"[^}]*?\n    \}', "", text)
    ids = [eid for eid, _ in design()]
    try:
        from scripts.generate_scaling_deep_dive import render_wave
    except ModuleNotFoundError:
        from generate_scaling_deep_dive import render_wave
    block = render_wave(
        dict(
            platform="l40s",
            wave=WAVE,
            name="L40S S7 real-text acceptance",
            experiment_ids=ids,
            required_trials=1,
            comparability="exact",
        )
    )
    marker = re.search(r'    \{\n      "platform": "l40s",\n      "wave": "topology-recovery-canary', text)
    if marker is None:
        raise ValueError("Cannot locate topology recovery wave section")
    text = text[: marker.start()] + block + ",\n" + text[marker.start() :]
    json.loads(text)
    return path, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path, text = expected_matrix()
    files = {path: text}
    files.update(
        {
            Path(CONFIG_ROOT) / f"{eid}_{data['experiment']}.json": json.dumps(data, indent=2) + "\n"
            for eid, data in design()
        }
    )
    if args.check:
        stale = [str(p) for p, content in files.items() if not p.exists() or p.read_text() != content]
        if stale:
            raise SystemExit("S7 acceptance files out of date: " + ", ".join(stale))
    else:
        for p, content in files.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    print("S7 acceptance: 8 fresh real-text canaries; no historical configs changed")


if __name__ == "__main__":
    main()
