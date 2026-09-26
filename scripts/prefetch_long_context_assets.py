"""Populate the Qwen/WikiText cache before reserving a long-context GPU job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpc_ai_perf.config.long_context import CONTEXT_IDS
from hpc_ai_perf.config.lowbit import BASE_MODEL, BASE_REVISION, DATASET, DATASET_REVISION, digest
from hpc_ai_perf.workloads.lowbit import text_windows, validate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-only", action="store_true", help="Validate the model/data contract without downloading weights"
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    config_file = hf_hub_download(BASE_MODEL, "config.json", revision=BASE_REVISION)
    model_config = json.loads(Path(config_file).read_text())
    validate_artifact("bf16", model_config)
    if not args.metadata_only:
        snapshot_download(
            BASE_MODEL,
            revision=BASE_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    dataset = load_dataset(DATASET, "wikitext-2-raw-v1", revision=DATASET_REVISION, split="validation")
    windows = {str(length): text_windows(dataset["text"], tokenizer, length, 8) for length in sorted(CONTEXT_IDS)}
    print(
        json.dumps(
            {
                "weights_downloaded": not args.metadata_only,
                "model": BASE_MODEL,
                "model_revision": BASE_REVISION,
                "model_config_sha256": digest(model_config),
                "dataset": DATASET,
                "dataset_revision": DATASET_REVISION,
                "validation_windows": {
                    length: {"windows": len(tokens), "tokens_per_window": int(length), "tokens_sha256": digest(tokens)}
                    for length, tokens in windows.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
