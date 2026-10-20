"""Check the pinned Qwen7B cache; optionally prefetch on the login node."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--prefetch", action="store_true", help="Allow downloads; run only on the login node")
args = parser.parse_args()
config = json.loads(next(Path("configs/inference").glob("E301_*.json")).read_text())["workload"]
snapshot = Path(
    snapshot_download(
        repo_id=config["model"],
        revision=config["model_revision"],
        local_files_only=not args.prefetch,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
)
required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
index = snapshot / "model.safetensors.index.json"
if index.is_file():
    required += sorted(set(json.loads(index.read_text())["weight_map"].values()))
else:
    required += ["model.safetensors"]
missing = [name for name in required if not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0]
if missing:
    raise SystemExit("Missing cached inputs: " + ", ".join(missing))
print(f"Pinned Qwen7B cache ready: {snapshot}")
