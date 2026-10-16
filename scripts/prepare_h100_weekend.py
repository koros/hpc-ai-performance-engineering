"""Prepare H100 inspection scripts and a pinned asset inventory without submitting jobs.

Run with PYTHONPATH=src. Prefetch is intended for the HPC login node; ordinary
preparation requires no CUDA, Slurm binaries, Hugging Face packages or network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from hpc_ai_perf.cli import _render_platform_slurm
from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.platform import load_platform_conditions
from hpc_ai_perf.validation.capabilities import capability_issues
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256

WAVES = (
    "scaling-dd-canary",
    "tinyllama-tp-long-prompt",
    "tinyllama-tp-prompt-sweep",
    "tinyllama-tp-concurrency-layout",
    "tinyllama-tp-output-length",
    "qwen7b-update-controls-2gpu",
    "qwen7b-tp-serving",
)
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"]


def asset_inventory(waves: list[str]) -> dict:
    entries = {entry.matrix_id: entry for entry in discover_experiment_configs()}
    selected = set()
    wave_rows = {}
    for wave in waves:
        conditions = load_platform_conditions("platforms/execution_matrix.json", "h100", wave=wave)
        if not conditions:
            raise ValueError(f"Unknown or empty H100 wave: {wave}")
        wave_rows[wave] = [{"id": c.matrix_id, "trials": c.required_trials} for c in conditions]
        selected.update(c.matrix_id for c in conditions)
    # Include referenced controls in the cache inventory, without submitting
    # historical controls or implying their old runtime is automatically matched.
    pending = list(selected)
    while pending:
        config = entries[pending.pop()].config
        refs = list(config.metadata.get("source_experiment_ids", []))
        control = config.workload.parameters.get("comparison_control_id")
        if control:
            refs.append(control)
        for reference in refs:
            if reference not in entries:
                raise ValueError(f"Missing comparison control: {reference}")
            if reference not in selected:
                selected.add(reference)
                pending.append(reference)
    assets: dict[tuple[str, str], list[str]] = {}
    datasets: dict[tuple[str, str, str], list[str]] = {}
    hashes = {}
    for eid in sorted(selected):
        entry = entries[eid]
        config = entry.config
        problems = capability_issues(config)
        if problems:
            raise ValueError(f"{eid}: {'; '.join(problems)}")
        revision = config.workload.model_revision
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{eid}: requires an immutable model revision")
        assets.setdefault((config.workload.model, revision), []).append(eid)
        params = config.workload.parameters
        if params.get("input_protocol") == "wikitext103_global_batches_v1":
            key = (config.workload.dataset, params["dataset_revision"], params["dataset_subset"])
            datasets.setdefault(key, []).append(eid)
        hashes[eid] = hashlib.sha256(entry.path.read_bytes()).hexdigest()
    return {
        "platform": "h100",
        "reserved_gpus": 4,
        "reserved_hours": 15,
        "inspection_only": True,
        "scaling_runtime_sha256": scaling_runtime_sha256(),
        "waves": wave_rows,
        "source_config_sha256": hashes,
        "models": [{"model": m, "revision": r, "conditions": ids} for (m, r), ids in sorted(assets.items())],
        "datasets": [
            {"dataset": d, "revision": r, "subset": s, "conditions": ids} for (d, r, s), ids in sorted(datasets.items())
        ],
    }


def cache_models(manifest: dict, *, offline: bool) -> None:
    from huggingface_hub import snapshot_download

    for asset in manifest["models"]:
        root = Path(
            snapshot_download(
                asset["model"],
                revision=asset["revision"],
                allow_patterns=PATTERNS,
                local_files_only=offline,
            )
        )
        if not (root / "config.json").is_file():
            raise ValueError(f"{asset['model']}: cached config.json is missing")
        tokenizer_files = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt")
        if not any((root / name).is_file() for name in tokenizer_files) and not (
            (root / "vocab.json").is_file() and (root / "merges.txt").is_file()
        ):
            raise ValueError(f"{asset['model']}: cached tokenizer vocabulary is missing")
        indexes = list(root.glob("*.safetensors.index.json"))
        if indexes:
            for index in indexes:
                names = set(json.loads(index.read_text())["weight_map"].values())
                if not names or any(not (root / name).is_file() for name in names):
                    raise ValueError(f"{asset['model']}: cached weight shards are incomplete")
        elif not list(root.glob("*.safetensors")):
            raise ValueError(f"{asset['model']}: cached safetensors weights are missing")
        print(f"Model snapshot present: {asset['model']} @ {asset['revision']}")
    if manifest["datasets"]:
        print(
            "Real-text canaries also require make prefetch-scaling-deep-dive-assets; model checks do not certify data."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", action="append", choices=WAVES, help="Repeat to narrow preparation")
    parser.add_argument("--output-dir", type=Path, default=Path("build/weekend_h100"))
    parser.add_argument("--prefetch", action="store_true", help="Download pinned model snapshots on the login node")
    parser.add_argument(
        "--verify-model-cache", action="store_true", help="Require complete cached model weights offline"
    )
    args = parser.parse_args()
    if args.prefetch and args.verify_model_cache:
        parser.error("Choose prefetch or offline model-cache verification")
    waves = args.wave or list(WAVES)
    manifest = asset_inventory(waves)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ASSETS.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prefetch or args.verify_model_cache:
        cache_models(manifest, offline=args.verify_model_cache)
        return
    for wave in waves:
        serving = wave.startswith("tinyllama-") or wave == "qwen7b-tp-serving"
        setup = ["source slurm/modules_for_hpc_ai"]
        if serving:
            setup.append("unset PYTHONPATH PYTHONHOME")
        setup.append(
            'source "${HPC_AI_WEEKEND_VLLM_ACTIVATE:?Set the certified vLLM activation path}"'
            if serving
            else "source .venv/bin/activate"
        )
        root = args.output_dir / wave
        _render_platform_slurm(
            platform_profile="platforms/h100.json",
            config_root="configs",
            execution_matrix="platforms/execution_matrix.json",
            wave=wave,
            experiment_ids=None,
            config_output_dir=str(root / "configs"),
            output_dir=str(root / "scripts"),
            run_output_dir=str(root / "results"),
            log_dir=str(root / "logs"),
            profile_path=None,
            partition=None,
            account=None,
            time_limit=None,
            cpus_per_task=None,
            modules=[],
            env_values=[],
            setup_commands_override=setup,
        )
    print(f"Prepared {len(waves)} H100 wave previews and pinned asset inventory at {args.output_dir}")
    print(
        "No jobs submitted. Use canonical immutable wave submission after cluster preflight; previews are not bundles."
    )


if __name__ == "__main__":
    main()
