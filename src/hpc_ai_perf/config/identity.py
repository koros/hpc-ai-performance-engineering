"""Stable identities for resolved experiment configurations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from hpc_ai_perf.config.experiment import ExperimentConfig


def canonical_config_bytes(config: ExperimentConfig | Mapping[str, Any]) -> bytes:
    """Serialize a config exactly once for result and bundle provenance.

    File formatting is not an experiment identity: a resolved JSON config can be
    indented differently without changing the workload that will run.  Bundle
    manifests and result records therefore share this canonical representation,
    while the manifest separately stores a raw file checksum for tamper checks.
    """
    payload = config.to_dict() if isinstance(config, ExperimentConfig) else dict(config)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def config_sha256(config: ExperimentConfig | Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 identity for one resolved config."""
    return hashlib.sha256(canonical_config_bytes(config)).hexdigest()
