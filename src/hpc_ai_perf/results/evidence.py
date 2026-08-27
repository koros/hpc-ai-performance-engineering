from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hpc_ai_perf.validation import validate_result_file, validate_result_records

MAX_EVIDENCE_BYTES = 5 * 1024 * 1024
SENSITIVE_KEY_TOKENS = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
    "token",
}
_MATRIX_ID = re.compile(r"^E[0-9]{3}$")


@dataclass(frozen=True)
class EvidenceImportSummary:
    """Files added or retained by one non-destructive evidence import."""

    imported: tuple[Path, ...]
    unchanged: tuple[Path, ...]
    manifests: tuple[Path, ...]


def import_result_evidence(
    source: str | Path,
    result_root: str | Path = "results/raw",
    *,
    max_file_bytes: int = MAX_EVIDENCE_BYTES,
) -> EvidenceImportSummary:
    """Import immutable matrix JSON evidence into platform-qualified directories."""
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Evidence source does not exist: {source_path}")
    candidates = [source_path] if source_path.is_file() else sorted(source_path.rglob("*.json"))
    root = Path(result_root)
    prepared: list[tuple[Path, bytes, str]] = []
    imported: list[Path] = []
    unchanged: list[Path] = []
    affected_platforms: set[str] = set()

    for candidate in candidates:
        if candidate.name == "SHA256SUMS":
            continue
        if candidate.stat().st_size > max_file_bytes:
            raise ValueError(f"Evidence file exceeds {max_file_bytes} bytes: {candidate}")
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid evidence JSON: {candidate}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Evidence record must be a JSON object: {candidate}")
        experiment_id = str(data.get("experiment_id", ""))
        if not _MATRIX_ID.fullmatch(experiment_id):
            continue
        security_issue = _sensitive_key_path(data)
        if security_issue:
            raise ValueError(f"Sensitive field `{security_issue}` is not allowed in evidence: {candidate}")
        validation_issues = validate_result_file(candidate)
        if validation_issues:
            messages = "; ".join(issue.message for issue in validation_issues)
            raise ValueError(f"Evidence validation failed for {candidate}: {messages}")
        platform_id = infer_evidence_platform(data)
        target = root / platform_id / experiment_id / candidate.name
        source_bytes = candidate.read_bytes()
        if target.exists() and _sha256_bytes(target.read_bytes()) != _sha256_bytes(source_bytes):
            raise FileExistsError(f"Refusing to replace different immutable evidence: {target}")
        prepared.append((target, source_bytes, platform_id))
        affected_platforms.add(platform_id)

    if not prepared:
        raise ValueError(f"No E001-E200 result records found under evidence source: {source_path}")
    for target, source_bytes, _ in prepared:
        if target.exists():
            unchanged.append(target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(target, source_bytes)
        imported.append(target)
    manifests = [write_evidence_manifest(root / platform_id) for platform_id in sorted(affected_platforms)]
    return EvidenceImportSummary(tuple(imported), tuple(unchanged), tuple(manifests))


def infer_evidence_platform(record: dict[str, Any]) -> str:
    """Infer one unambiguous L40S/H100 identity from recorded result metadata."""
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    hardware = config.get("hardware") if isinstance(config.get("hardware"), dict) else {}
    environment = record.get("environment") if isinstance(record.get("environment"), dict) else {}
    signals = [
        record.get("platform_id"),
        metadata.get("execution_platform"),
        hardware.get("platform"),
        hardware.get("gpu_model"),
        environment.get("observed_gpu_model"),
    ]
    platforms: set[str] = set()
    for signal in signals:
        text = str(signal or "").lower()
        if "l40" in text:
            platforms.add("l40s")
        if "h100" in text:
            platforms.add("h100")
    if len(platforms) != 1:
        observed = ", ".join(sorted(platforms)) or "none"
        raise ValueError(f"Evidence must identify exactly one L40S/H100 platform; observed: {observed}")
    return next(iter(platforms))


def write_evidence_manifest(platform_dir: str | Path) -> Path:
    """Write a deterministic SHA256SUMS manifest for one platform evidence tree."""
    root = Path(platform_dir)
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for result_path in sorted(root.rglob("*.json")):
        relative_path = result_path.relative_to(root).as_posix()
        lines.append(f"{_sha256_bytes(result_path.read_bytes())}  {relative_path}")
    manifest_path = root / "SHA256SUMS"
    payload = ("\n".join(lines) + "\n").encode("utf-8") if lines else b""
    _atomic_write_bytes(manifest_path, payload)
    return manifest_path


def verify_result_evidence(result_root: str | Path = "results/raw") -> list[str]:
    """Return validation, checksum, and manifest-coverage problems for committed evidence."""
    root = Path(result_root)
    problems = [f"{issue.path}: {issue.message}" for issue in validate_result_records(root)]
    for platform_id in ["l40s", "h100"]:
        platform_dir = root / platform_id
        if not platform_dir.exists():
            continue
        manifest_path = platform_dir / "SHA256SUMS"
        if not manifest_path.exists():
            problems.append(f"{manifest_path}: missing evidence checksum manifest")
            continue
        expected: dict[str, str] = {}
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split("  ", 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                problems.append(f"{manifest_path}:{line_number}: malformed checksum line")
                continue
            if parts[1] in expected:
                problems.append(f"{manifest_path}:{line_number}: duplicate manifest path: {parts[1]}")
            expected[parts[1]] = parts[0]
        actual_paths = {path.relative_to(platform_dir).as_posix(): path for path in platform_dir.rglob("*.json")}
        for relative_path in sorted(set(actual_paths) - set(expected)):
            problems.append(f"{actual_paths[relative_path]}: result is missing from SHA256SUMS")
        for relative_path in sorted(set(expected) - set(actual_paths)):
            problems.append(f"{manifest_path}: listed result is missing: {relative_path}")
        for relative_path in sorted(set(expected) & set(actual_paths)):
            result_path = actual_paths[relative_path]
            actual_hash = _sha256_bytes(result_path.read_bytes())
            if actual_hash != expected[relative_path]:
                problems.append(f"{result_path}: SHA-256 does not match SHA256SUMS")
            try:
                record = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                try:
                    inferred_platform = infer_evidence_platform(record)
                except ValueError as exc:
                    problems.append(f"{result_path}: {exc}")
                    continue
                if inferred_platform != platform_id:
                    problems.append(f"{result_path}: evidence platform does not match parent directory")
                experiment_id = str(record.get("experiment_id", ""))
                relative_parts = Path(relative_path).parts
                if not relative_parts or relative_parts[0] != experiment_id:
                    problems.append(f"{result_path}: experiment_id does not match parent directory")
                security_issue = _sensitive_key_path(record)
                if security_issue:
                    problems.append(f"{result_path}: sensitive field is present: {security_issue}")
    return problems


def _sensitive_key_path(value: object, prefix: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            key_path = f"{prefix}.{key}" if prefix else str(key)
            key_parts = set(normalized.split("_"))
            credential_token = "token" in key_parts and bool(key_parts & {"access", "api", "auth", "hf", "refresh"})
            if (
                normalized in SENSITIVE_KEY_TOKENS
                or "password" in key_parts
                or "secret" in key_parts
                or credential_token
            ):
                return key_path
            found = _sensitive_key_path(item, key_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _sensitive_key_path(item, f"{prefix}[{index}]")
            if found:
                return found
    return None


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file_obj:
        temporary_path = Path(file_obj.name)
        file_obj.write(payload)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary_path, path)
