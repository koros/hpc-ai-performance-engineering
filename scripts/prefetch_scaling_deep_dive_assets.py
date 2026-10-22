#!/usr/bin/env python3
"""Populate the model and dataset cache for the scaling deep dive before reserving GPUs.

Downloads the pinned weights and tokenizers of every deep-dive workload and the
pinned WikiText-103 training split, then checks that the corpus can supply the
largest real-text condition (most documents and most packed tokens) per model.
"""

from __future__ import annotations

import argparse
import json

from hpc_ai_perf.config.scaling import REAL_TEXT_DATASET, REAL_TEXT_REVISION, REAL_TEXT_SUBSET
from hpc_ai_perf.config.scaling_deep_dive import WORKLOADS, design
from hpc_ai_perf.workloads.scaling import MIN_DOCUMENT_TOKENS, wikitext_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true", help="Fetch configs and tokenizers, not weights")
    parser.add_argument("--real-text-only", action="store_true", help="Fetch only S7's GPT-2 and TinyLlama assets")
    parser.add_argument("--platform", choices=("h100", "l40s"), help="Restrict corpus sizing to one platform")
    parser.add_argument("--offline", action="store_true", help="Verify using only existing cached assets")
    args = parser.parse_args()
    if args.offline:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    conditions = [
        c for _, c, _ in design() if c.text_batching and (not args.platform or args.platform in c.run_platforms)
    ]
    selected = {c.workload for c in conditions} if args.real_text_only else set(WORKLOADS)
    patterns = ["*.json", "*.txt", "*.model"] + ([] if args.metadata_only else ["*.safetensors"])
    for name in sorted(selected):
        spec = WORKLOADS[name]
        snapshot_download(
            spec["model"], revision=spec["revision"], allow_patterns=patterns, local_files_only=args.offline
        )

    data = load_dataset(REAL_TEXT_DATASET, REAL_TEXT_SUBSET, revision=REAL_TEXT_REVISION, split="train")
    documents = wikitext_documents(data["text"])
    report = {"dataset_revision": REAL_TEXT_REVISION, "documents": len(documents), "workloads": {}}
    for workload in sorted({c.workload for c in conditions}):
        shapes = [c for c in conditions if c.workload == workload]
        need_docs = max((10 + c.updates) * c.global_batch for c in shapes)
        need_tokens = max((10 + c.updates) * c.global_batch * c.sequence_length for c in shapes)
        spec = WORKLOADS[workload]
        tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
        usable, tokens = 0, 0
        for text in documents:
            count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if count >= MIN_DOCUMENT_TOKENS:
                usable, tokens = usable + 1, tokens + count + 1
            if usable >= need_docs and tokens >= need_tokens:
                break
        if usable < need_docs or tokens < need_tokens:
            raise SystemExit(
                f"{workload}: corpus supplies {usable} docs/{tokens} tokens; needs {need_docs}/{need_tokens}"
            )
        report["workloads"][workload] = {"documents_needed": need_docs, "packed_tokens_needed": need_tokens}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
