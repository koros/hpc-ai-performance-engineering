"""Refresh newer catalog conditions and inference findings from checksum-verified raw evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from html import escape
from pathlib import Path

try:
    from scripts.update_dashboard_tinyllama_strategy import replace_embedded, totals
except ModuleNotFoundError:  # Direct script execution adds scripts/, not the repository root.
    from update_dashboard_tinyllama_strategy import replace_embedded, totals

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "strategy",
    "precision",
    "gpus",
    "batch_size",
    "global_batch_size",
    "per_gpu_batch_size",
    "sequence_length",
    "prompt_length",
    "concurrent_requests",
    "max_new_tokens",
    "framework",
    "serving_layout",
    "tensor_parallel_size",
    "replica_count",
    "gradient_accumulation_steps",
    "activation_checkpointing",
    "pinned_memory",
    "num_workers",
    "prefetch_factor",
    "bucket_cap_mb",
    "ddp_comm_hook",
    "gradient_as_bucket_view",
    "static_graph",
    "sequence_split",
    "data_source",
    "text_batching_strategy",
    "max_model_len",
    "enable_prefix_caching",
)
METRICS = (
    "throughput_tokens_sec",
    "throughput_samples_sec",
    "memory_used_gb",
    "runtime_seconds",
    "latency_p50_ms",
    "latency_p95_ms",
    "time_to_first_token_p95_ms",
    "time_per_output_token_p95_ms",
    "power_draw_watts",
)


def current_records():
    """Keep runtime/config cohorts separate; use latest cohort per condition/platform."""
    groups = defaultdict(lambda: defaultdict(list))
    for platform in ("l40s", "h100"):
        root = ROOT / "results/raw" / platform
        checksums = {}
        for line in (root / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            checksums[name.lstrip("*")] = digest
        for path in sorted(root.glob("E*/*.json")):
            r = json.loads(path.read_text())
            if int(r.get("experiment_id", "E0")[1:]) < 294 or r.get("status") != "completed" or r.get("error"):
                continue
            if checksums.get(path.relative_to(root).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"Checksum mismatch: {path}")
            env = r.get("environment") or {}
            m = r.get("metrics") or {}
            cohort = (
                r.get("config_sha256"),
                m.get("serving_runtime_sha256"),
                env.get("torch_version"),
                env.get("vllm_version"),
            )
            groups[(r["experiment_id"], platform)][cohort].append(r)
    return {
        key: max(cohorts.values(), key=lambda rs: max(r["completed_at"] for r in rs)) for key, cohorts in groups.items()
    }


def measured(rows):
    result = {"n": len(rows), "gpu_model": rows[0]["config"]["hardware"]["gpu_model"]}
    for field in METRICS:
        values = [r.get("metrics", {}).get(field, r.get(field)) for r in rows]
        if all(isinstance(v, int | float) and math.isfinite(v) for v in values):
            result[field] = statistics.mean(values)
    energies = [r["metrics"].get("energy_joules_measured_region") for r in rows]
    if all(isinstance(v, int | float) and math.isfinite(v) for v in energies):
        result["energy_joules"] = statistics.mean(energies)
        tokens = [r["metrics"].get("generated_tokens") for r in rows]
        if all(isinstance(t, int | float) and math.isfinite(t) and t > 0 for t in tokens):
            result["joules_per_output_token"] = statistics.mean(e / t for e, t in zip(energies, tokens, strict=True))
    result["energy_scope"] = "measured_region" if "energy_joules" in result else "unavailable"
    result["source_run_ids"] = [r["run_id"] for r in rows]
    result["serving_runtime_sha256"] = rows[0]["metrics"].get("serving_runtime_sha256")
    result["throughput_min"] = min(r["metrics"].get("throughput_tokens_sec", 0) for r in rows)
    result["throughput_max"] = max(r["metrics"].get("throughput_tokens_sec", 0) for r in rows)
    return result


def build_catalog():
    catalog = json.loads((ROOT / "dashboard/experiments-data.json").read_text())
    details = json.loads((ROOT / "dashboard/experiment-details.json").read_text())
    records = current_records()
    policy = defaultdict(dict)
    for platform in ("l40s", "h100"):
        for c in load_platform_conditions(ROOT / "platforms/execution_matrix.json", platform):
            policy[c.matrix_id][platform] = c.required_trials
    # Rebuild only the new groups; keep the original narrative snapshot's grouping.
    for theme in catalog["themes"]:
        theme["phases"] = [p for p in theme["phases"] if not p.get("live_catalog")]
    experiments = catalog["experiments"]
    added = defaultdict(lambda: defaultdict(list))
    for entry in discover_experiment_configs():
        eid = entry.matrix_id
        if int(eid[1:]) < 294:
            continue
        c = entry.config.to_dict()
        p = c["workload"]["parameters"]
        meta = c["metadata"]
        mode = c["mode"]
        theme_id = "serving" if mode == "inference" else "scaling"
        theme = next(t for t in catalog["themes"] if t["id"] == theme_id)
        phase = meta.get("platform_wave") or meta.get("phase") or "Additional studies"
        # Avoid implicit deltas between different models, request shapes or roles.
        cluster = (
            c["workload"]["name"],
            p.get("prompt_length"),
            p.get("concurrent_requests"),
            p.get("max_new_tokens"),
            meta.get("comparison_scope"),
            p.get("inference_study_type"),
        )
        platforms = set(policy[eid]) | {pl for (key, pl) in records if key == eid}
        if not platforms:
            hint = c["hardware"].get("platform", "")
            platforms = (
                {hint.removeprefix("lovelace_")} if hint in ("lovelace_l40s", "lovelace_h100") else {"l40s", "h100"}
            )
        observed = {pl: measured(records[(eid, pl)]) for pl in sorted(platforms) if (eid, pl) in records}
        complete = any(m["n"] >= policy[eid].get(pl, 3) for pl, m in observed.items())
        for pl, m in observed.items():
            m["coverage_status"] = "Completed" if m["n"] >= policy[eid].get(pl, 3) else "Configured"
        experiment = dict(
            id=eid,
            status="Completed" if complete else "Configured",
            phase=phase,
            theme_id=theme_id,
            theme_label=theme["label"],
            workload=c["workload"]["name"],
            workload_label=c["workload"]["model"].split("/")[-1],
            mode=mode,
            model=c["workload"]["model"],
            comparability=meta.get("comparability", "exact"),
            platforms=observed,
            platform_list=sorted(platforms),
            research_questions=meta.get("research_questions", []),
            objective=meta.get("objective", ""),
            priority=meta.get("priority", "Optional"),
            gpu_hours_estimate=meta.get("gpu_hours_estimate"),
            completed_at=max((r["completed_at"] for pl in platforms for r in records.get((eid, pl), [])), default=None),
            notes="Acceptance diagnostic; excluded from performance findings." if "canary" in phase else None,
        )
        for key in FIELDS:
            value = p.get(key, c.get("distributed", {}).get(key, c.get("runtime", {}).get(key)))
            if key == "gpus":
                value = c["hardware"]["gpus"]
            experiment[key] = str(value) if value is not None else None
        experiment["runtime_seconds"] = (
            statistics.mean(m["runtime_seconds"] for m in observed.values()) if observed else None
        )
        experiments[eid] = experiment
        experiment["model_revision"] = c["workload"].get("model_revision")
        experiment["comparison_scope"] = p.get("comparison_scope")
        detail = {k: v for k, v in experiment.items() if k not in ("platforms", "theme_id", "theme_label")}
        detail["platforms"] = {f"lovelace_{pl}": dict(m, energy_j=m.get("energy_joules")) for pl, m in observed.items()}
        details[eid] = detail
        added[(theme_id, phase)][cluster].append(eid)
    for (theme_id, phase), clusters in added.items():
        theme = next(t for t in catalog["themes"] if t["id"] == theme_id)
        rendered = []
        for number, (_, ids) in enumerate(clusters.items()):
            members = [experiments[eid] for eid in ids]
            constant, varied = {}, {}
            for field in FIELDS + ("model", "mode", "platform_list", "comparability"):
                values = list(
                    {json.dumps(m.get(field)): m.get(field) for m in members if m.get(field) is not None}.values()
                )
                if len(values) == 1:
                    constant[field] = values[0]
                elif values:
                    varied[field] = values
            rendered.append(
                dict(
                    key=f"live-{phase}-{number}",
                    label=members[0]["workload_label"],
                    totals=totals(ids, experiments),
                    constant=constant,
                    varied=varied,
                    experiment_ids=ids,
                )
            )
        ids = [eid for cluster in rendered for eid in cluster["experiment_ids"]]
        theme["phases"].append(
            dict(
                id="live-" + re.sub(r"[^a-z0-9]+", "-", phase.lower()),
                label=phase.replace("-", " ")
                .replace("qwen7b", "Qwen7B")
                .replace("scaling dd", "Training deep dive")
                .replace("inference crossover", "Inference layouts"),
                live_catalog=True,
                totals=totals(ids, experiments),
                clusters=rendered,
            )
        )
    catalog["meta"].update(totals(list(experiments), experiments))
    catalog["meta"]["generated_at"] = max(r["completed_at"][:10] for rows in records.values() for r in rows)
    for key in ("precision", "strategy", "mode"):
        catalog["filter_options"][key] = sorted({e[key] for e in experiments.values() if e.get(key)})
    catalog["filter_options"]["workload"] = sorted(
        {e["workload"]: e["workload_label"] for e in experiments.values()}.items()
    )
    return catalog, details


def inference_section(catalog):
    experiments = catalog["experiments"]
    groups = defaultdict(list)
    for eid, e in experiments.items():
        if int(eid[1:]) < 715 or "canary" in e["phase"] or e["status"] != "Completed":
            continue
        for platform, m in e["platforms"].items():
            if m["coverage_status"] != "Completed":
                continue
            groups[
                (
                    e["model"],
                    platform,
                    e["prompt_length"],
                    e["concurrent_requests"],
                    m.get("serving_runtime_sha256"),
                    e.get("model_revision"),
                    e.get("comparison_scope"),
                )
            ].append((eid, e, m))
    parts = [
        "<!-- LIVE_INFERENCE_START -->",
        '<section id="inference-parallelization">',
        '<p class="section-kicker">Latest measured studies</p>',
        "<h2>Inference parallelization: speed, first-token delay and energy</h2>",
        "<p>More GPUs per model is not always faster. These comparisons hold the node GPU budget fixed "
        "and split it between model shards and independent serving replicas. TinyLlama favoured single-GPU replicas; "
        "Qwen7B favoured TP2 throughput, with a demand-dependent energy trade-off.</p>",
        "<p>TP2 × DP4 means four replicas, each spread over two GPUs. All results below use BF16, "
        "128 fixed output tokens and the recorded serving runtime. Acceptance trials are excluded. "
        "Values are means across independent trials; throughput ranges show minimum–maximum across trials. "
        "p95 values are means of within-trial percentiles.</p>",
        '<p><a href="index.html">Browse every condition and its platform status →</a></p>',
        "<h3>New training and acceptance coverage</h3>",
        "<p>The catalog now includes every checked-in E-number condition. Completed means at least one "
        "platform has the required collected trials; it does not imply both platforms are complete. "
        "Acceptance is displayed separately from performance evidence.</p>",
        '<div class="table-wrap"><table class="nerd-table"><thead><tr><th>L40S study</th>'
        "<th>Completed conditions</th><th>Selected-cohort trials</th><th>Source range</th></tr></thead><tbody>",
    ]
    for label, start, end in [
        ("Accumulation", 386, 433),
        ("Sequence split", 434, 505),
        ("Model size", 506, 533),
        ("DDP communication", 534, 578),
        ("Training stacks", 579, 607),
        ("Placement/topology", 608, 633),
        ("Real text", 638, 669),
        ("Precision", 670, 697),
    ]:
        members = [
            experiments[f"E{i}"] for i in range(start, end + 1) if "l40s" in experiments[f"E{i}"]["platform_list"]
        ]
        done = sum(e["platforms"].get("l40s", {}).get("coverage_status") == "Completed" for e in members)
        count = sum(e["platforms"].get("l40s", {}).get("n", 0) for e in members)
        parts.append(
            f"<tr><td>{label}</td><td>{done}/{len(members)}</td><td>{count}</td><td>E{start}@l40s–E{end}@l40s</td></tr>"
        )
    parts.append("</tbody></table></div>")
    for (model, platform, prompt, concurrency, *_cohort), rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row[1]["tensor_parallel_size"]))
        parts.append(
            f"<h3>{escape(model.split('/')[-1])} · {platform.upper()} · "
            f"prompt {prompt} · concurrency {concurrency}</h3>"
        )
        maximum = max(m["throughput_tokens_sec"] for _, _, m in rows)
        parts.append('<div class="live-chart" role="group" aria-label="Output throughput by layout">')
        for _, e, m in rows:
            width = 100 * m["throughput_tokens_sec"] / maximum
            parts.append(
                f'<div class="live-bar-row"><span>{escape(e["serving_layout"])}</span>'
                f'<div class="live-bar-track"><div class="live-bar" style="width:{width:.2f}%"></div></div>'
                f"<strong>{m['throughput_tokens_sec']:,.1f} tok/s</strong></div>"
            )
        parts.append(
            '</div><div class="table-wrap"><table class="nerd-table"><thead><tr>'
            "<th>Layout</th><th>Trials</th><th>Output tok/s</th><th>p95 TTFT (ms)</th>"
            "<th>p95 time/token (ms)</th><th>J/output token</th><th>Source</th></tr></thead><tbody>"
        )
        for eid, e, m in rows:
            energy = m.get("joules_per_output_token")
            parts.append(
                f"<tr><td>{escape(e['serving_layout'])}</td><td>{m['n']}</td>"
                f"<td>{m['throughput_tokens_sec']:,.1f}<br><small>{m['throughput_min']:,.1f}–{m['throughput_max']:,.1f}</small></td><td>{m['time_to_first_token_p95_ms']:.1f}</td>"
                f"<td>{m['time_per_output_token_p95_ms']:.2f}</td>"
                f"<td>{energy:.2f}</td><td>{eid}@{platform}</td></tr>"
            )
        parts.append("</tbody></table></div>")
    parts += [
        "<h3>How to interpret these findings</h3>",
        "<p>Qwen7B TP2 and TP4 had nearly equal throughput at concurrency 1; TP2 used less energy per output "
        "token. At concurrency 8, TP2 led throughput while TP4 used less energy per output token. "
        "Only collected performance conditions appear here; configured follow-ups remain in the catalog.</p>",
        "<p>Energy is integrated across all allocated GPUs over the measured region. Memory totals reflect "
        "vLLM cache reservation and do not isolate model weights. Request timestamps exclude HTTP/network "
        "and some parent-router overhead. Synthetic closed rounds do not establish production queueing or an SLO. "
        "No direct NVLink/PCIe traffic attribution is available. "
        "Cross-platform comparisons require matched controls.</p>",
        "</section><!-- LIVE_INFERENCE_END -->",
    ]
    return "\n".join(parts)


def update():
    catalog, details = build_catalog()

    def compact(data):
        return json.dumps(data, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")

    index = ROOT / "dashboard/index.html"
    analysis = ROOT / "dashboard/analysis.html"
    outer = replace_embedded(index.read_text(), "DATA", compact(catalog))
    inner = replace_embedded(analysis.read_text(), "EXPERIMENT_DETAILS", compact(details))
    block = inference_section(catalog)
    inner = re.sub(r"<!-- LIVE_INFERENCE_START -->.*?<!-- LIVE_INFERENCE_END -->\n*", "", inner, flags=re.S)
    inner = inner.replace('  <section id="scaling">', block + '\n\n  <section id="scaling">', 1)
    total, completed = catalog["meta"]["total"], catalog["meta"]["completed"]
    inner = re.sub(
        r"var total = \d+, completed = \d+, pct =", f"var total = {total}, completed = {completed}, pct =", inner
    )
    inner = re.sub(r'(<div class="value" id="coverageKpi">)[^<]*', rf"\g<1>{completed} / {total}", inner)
    inner = re.sub(
        r"planned tests completed so far \(\d+%\)",
        f"planned conditions complete ({round(completed / total * 100)}%)",
        inner,
    )
    inner = inner.replace(
        "spanning two chip types and three AI models.",
        "covering the checked-in experiment matrix; latest collected inference evidence is shown first.",
    )
    for element, value in (
        ("coverageCompleted", completed),
        ("coverageTotal", total),
        ("coveragePct", f"{round(completed / total * 100)}%"),
    ):
        inner = re.sub(rf'(<b id="{element}">)[^<]*', rf"\g<1>{value}", inner)
    return {
        ROOT / "dashboard/experiments-data.json": compact(catalog) + "\n",
        ROOT / "dashboard/experiment-details.json": compact(details) + "\n",
        index: outer,
        analysis: inner,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = update()
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, text in expected.items() if p.read_text() != text]
        if stale:
            raise SystemExit("Stale dashboard results: " + ", ".join(stale))
        print("Dashboard matches collected results.")
    else:
        for p, text in expected.items():
            p.write_text(text)
        print("Updated catalog, details and inference analysis from raw evidence.")


if __name__ == "__main__":
    main()
