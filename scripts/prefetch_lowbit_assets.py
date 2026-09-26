"""Populate the immutable model/data cache before spending a GPU reservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc_ai_perf.config.lowbit import ARTIFACTS, BASE_MODEL, BASE_REVISION, DATASET, DATASET_REVISION, digest
from hpc_ai_perf.workloads.lowbit import text_windows, validate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only", action="store_true", help="Validate inputs without downloading model weights"
    )
    parser.add_argument("--quality-extension", action="store_true", help="Validate the 128-window quality study inputs")
    args = parser.parse_args()
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    artifacts = {}
    for method in ("bf16", "awq", "gptq"):
        model, revision, _ = ARTIFACTS[method]
        config_file = hf_hub_download(model, "config.json", revision=revision)
        config = json.loads(Path(config_file).read_text())
        validate_artifact(method, config)
        if not args.metadata_only:
            snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        artifacts[method] = {"model": model, "revision": revision, "config_sha256": digest(config)}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    inputs = {}
    quality_shape = ("test", 512, 128) if args.quality_extension else ("test", 256, 32)
    for split, length, count in (("validation", 512, 64), quality_shape):
        data = load_dataset(DATASET, "wikitext-2-raw-v1", revision=DATASET_REVISION, split=split)
        windows = text_windows(data["text"], tokenizer, length, count)
        inputs[split] = {"windows": len(windows), "tokens_per_window": length, "tokens_sha256": digest(windows)}
    print(
        json.dumps(
            {
                "weights_downloaded": not args.metadata_only,
                "artifacts": artifacts,
                "dataset_revision": DATASET_REVISION,
                "inputs": inputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
