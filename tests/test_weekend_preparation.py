import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest


def preparation_module():
    spec = importlib.util.spec_from_file_location("weekend_prep", "scripts/prepare_h100_weekend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_includes_controls_and_real_text_without_launching():
    module = preparation_module()
    inventory = module.asset_inventory(list(module.WAVES))
    assert len(inventory["waves"]["scaling-dd-canary"]) == 63
    assert inventory["reserved_gpus"] == 4
    assert "E194" in inventory["source_config_sha256"]
    assert len(inventory["datasets"]) == 1
    assert inventory["datasets"][0]["subset"] == "wikitext-103-raw-v1"
    assert all(len(a["revision"]) == 40 for a in inventory["models"])


def test_offline_model_check_rejects_a_partially_collected_snapshot(tmp_path, monkeypatch):
    module = preparation_module()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "shard-1.safetensors", "b": "shard-2.safetensors"}})
    )
    (tmp_path / "shard-1.safetensors").write_bytes(b"cached")
    calls = []

    def snapshot(model, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    manifest = {"models": [{"model": "example/model", "revision": "a" * 40}], "datasets": []}
    with pytest.raises(ValueError, match="weight shards are incomplete"):
        module.cache_models(manifest, offline=True)
    assert calls[0]["local_files_only"] is True
    (tmp_path / "shard-2.safetensors").write_bytes(b"cached")
    module.cache_models(manifest, offline=True)


def test_preview_serving_environment_is_explicit_and_no_submission_occurs(tmp_path, monkeypatch):
    module = preparation_module()
    calls = []
    monkeypatch.setattr(module, "_render_platform_slurm", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--wave",
            "tinyllama-tp-concurrency-layout",
            "--output-dir",
            str(tmp_path),
        ],
    )
    module.main()
    assert (tmp_path / "ASSETS.json").is_file()
    assert len(calls) == 1
    setup = calls[0]["setup_commands_override"]
    assert "unset PYTHONPATH PYTHONHOME" in setup
    assert "HPC_AI_WEEKEND_VLLM_ACTIVATE:?" in setup[-1]
