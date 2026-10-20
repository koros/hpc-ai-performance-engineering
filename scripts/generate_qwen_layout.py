"""Generate the independent inference-first matrix without changing old IDs."""

import argparse
import json
import re
from pathlib import Path

from hpc_ai_perf.config.qwen_layout import (
    CANARY_WAVE,
    CONFIG_ROOT,
    EXTENSION_CANARY_WAVE,
    EXTENSION_PERFORMANCE_WAVE,
    LOW_CONCURRENCY_CANARY_WAVE,
    LOW_CONCURRENCY_PERFORMANCE_WAVE,
    PERFORMANCE_WAVE,
    design,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = design()
    expected = {}
    for eid, platform, wave, data in rows:
        p = data["workload"]["parameters"]
        role = "canary" if wave in (CANARY_WAVE, EXTENSION_CANARY_WAVE, LOW_CONCURRENCY_CANARY_WAVE) else "performance"
        stem = f"{eid}_{platform}_{p['serving_layout']}_p{p['prompt_length']}_c{p['concurrent_requests']}_{role}"
        expected[Path(CONFIG_ROOT) / f"{stem}.json"] = json.dumps(data, indent=2) + "\n"
    matrix_path = Path("platforms/execution_matrix.json")
    text = matrix_path.read_text()
    text = re.sub(
        r'    \{\n      "platform": "[a-z0-9]+",\n      "wave": "qwen7b-layout-[^}]*?\n    \},\n',
        "",
        text,
    )
    blocks = []
    for platform in ("l40s",):
        for wave in (
            CANARY_WAVE,
            PERFORMANCE_WAVE,
            EXTENSION_CANARY_WAVE,
            EXTENSION_PERFORMANCE_WAVE,
            LOW_CONCURRENCY_CANARY_WAVE,
            LOW_CONCURRENCY_PERFORMANCE_WAVE,
        ):
            ids = [eid for eid, p, w, _ in rows if p == platform and w == wave]
            blocks.append(
                {
                    "platform": platform,
                    "wave": wave,
                    "name": f"{platform.upper()} equal-budget inference {wave}",
                    "experiment_ids": ids,
                    "required_trials": 1
                    if wave in (CANARY_WAVE, EXTENSION_CANARY_WAVE, LOW_CONCURRENCY_CANARY_WAVE)
                    else 3,
                    "comparability": "exact",
                }
            )
    rendered = []
    for block in blocks:
        ids = ", ".join(json.dumps(eid) for eid in block["experiment_ids"])
        rendered.append(
            "    {\n"
            f'      "platform": "{block["platform"]}",\n'
            f'      "wave": "{block["wave"]}",\n'
            f'      "name": {json.dumps(block["name"])},\n'
            f'      "experiment_ids": [{ids}],\n'
            f'      "required_trials": {block["required_trials"]},\n'
            '      "comparability": "exact"\n    }'
        )
    marker = re.search(r'    \{\n      "platform": "[a-z0-9]+",\n      "wave": "scaling-batch-canary', text)
    if marker is None:
        raise ValueError("Cannot locate the existing deep-dive wave section")
    expected[matrix_path] = text[: marker.start()] + ",\n".join(rendered) + ",\n" + text[marker.start() :]
    json.loads(expected[matrix_path])
    if args.check:
        stale = [str(p) for p, text in expected.items() if not p.is_file() or p.read_text() != text]
        if stale:
            raise SystemExit("Inference matrix drift: " + ", ".join(stale))
    else:
        for path, text in expected.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    print(
        f"Qwen layout design: {len(rows)} configs; 6 canaries and 27 performance trials "
        "(including completed original study)"
    )


if __name__ == "__main__":
    main()
