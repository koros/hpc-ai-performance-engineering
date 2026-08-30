"""Immutable, Slurm-safe submission bundles for platform experiment waves.

Jobs in a Slurm wave can begin long after they are submitted.  A bundle keeps
their code and resolved configuration independent from the checkout used to
import and commit results while they wait in the scheduler.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import config_sha256

_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MUTABLE_RESULT_PATHS = (
    "build/",
    "results/raw/",
    "results/processed/",
    "results/analysis/",
    "results/figures/",
    "slurm/generated/",
)
_EXECUTION_INPUT_PATHS = ("src/", "configs/", "platforms/", "slurm/")
_EXECUTION_INPUT_FILES = {"pyproject.toml"}


@dataclass(frozen=True)
class SubmissionBundle:
    """Filesystem locations and immutable identity for one platform-wave submission."""

    bundle_id: str
    root: Path
    source_repository: Path
    repo_dir: Path
    config_dir: Path
    script_dir: Path
    result_dir: Path
    log_dir: Path
    cache_dir: Path
    manifest_path: Path
    result_manifest_path: Path
    script_checksums_path: Path
    git_commit: str
    created_at: str


def prepare_submission_bundle(
    *,
    repository: str | Path,
    bundle_root: str | Path,
    bundle_id: str,
) -> SubmissionBundle:
    """Create an empty bundle and a detached worktree at the committed source revision.

    Only execution inputs must be clean.  Evidence and generated outputs are
    deliberately allowed to change in the primary checkout while a bundle is
    queued, because they are not copied into the snapshot.
    """
    if not _BUNDLE_ID.fullmatch(bundle_id):
        raise ValueError("bundle_id must use letters, numbers, dots, underscores, or hyphens")

    repo_dir = _git_repository_root(repository)
    _validate_execution_inputs_clean(repo_dir)
    git_commit = _git(repo_dir, "rev-parse", "HEAD")
    root = Path(bundle_root).expanduser().resolve() / bundle_id
    if root.exists():
        raise FileExistsError(f"Submission bundle already exists: {root}")

    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_git(repo_dir, "worktree", "add", "--detach", str(root / "repo"), git_commit)
        for directory in ("configs", "scripts", "results", "logs", "cache"):
            (root / directory).mkdir(parents=True, exist_ok=True)
    except Exception:
        _remove_new_bundle(repo_dir, root)
        raise

    created_at = datetime.now(UTC).isoformat()
    return SubmissionBundle(
        bundle_id=bundle_id,
        root=root,
        source_repository=repo_dir,
        repo_dir=root / "repo",
        config_dir=root / "configs",
        script_dir=root / "scripts",
        result_dir=root / "results",
        log_dir=root / "logs",
        cache_dir=root / "cache",
        manifest_path=root / "SUBMISSION_MANIFEST.json",
        result_manifest_path=root / "results" / "SUBMISSION_MANIFEST.json",
        script_checksums_path=root / "SCRIPT_SHA256SUMS",
        git_commit=git_commit,
        created_at=created_at,
    )


def remove_submission_bundle(bundle: SubmissionBundle) -> None:
    """Remove a bundle that failed before submission; never touches the primary checkout."""
    _remove_new_bundle(bundle.source_repository, bundle.root)


def write_submission_manifest(
    bundle: SubmissionBundle,
    *,
    platform_id: str,
    wave: str,
    input_files: dict[str, Path],
) -> dict[str, Any]:
    """Write the immutable provenance manifest after platform configs are frozen."""
    config_entries: dict[str, dict[str, Any]] = {}
    for config_path in sorted(bundle.config_dir.glob("E*.json")):
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Resolved config must be a JSON object: {config_path}")
        experiment_id = str(data.get("experiment_id", ""))
        if not re.fullmatch(r"E[0-9]{3}", experiment_id):
            raise ValueError(f"Resolved config has invalid experiment_id: {config_path}")
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        config_entries[experiment_id] = {
            "path": config_path.name,
            "file_sha256": _sha256_path(config_path),
            "config_sha256": config_sha256(data),
            "condition_id": str(metadata.get("condition_id", "")),
            "required_trials": int(metadata.get("required_trials", 1)),
        }
    if not config_entries:
        raise ValueError(f"No resolved configs found in submission bundle: {bundle.config_dir}")

    manifest = {
        "schema_version": "1.0",
        "bundle_id": bundle.bundle_id,
        "platform_id": platform_id,
        "wave": str(wave),
        "created_at": bundle.created_at,
        "repository": {
            "git_commit": bundle.git_commit,
            "snapshot_directory": "repo",
        },
        "configs": config_entries,
        "input_files": {
            name: {"path": str(path), "sha256": _sha256_path(path)} for name, path in sorted(input_files.items())
        },
        "environment_contract": capture_environment_contract(),
    }
    _write_json(bundle.manifest_path, manifest)
    _write_json(bundle.result_manifest_path, manifest)
    return manifest


def write_script_checksums(bundle: SubmissionBundle) -> Path:
    """Write reviewable hashes for the scripts produced by a submission bundle."""
    lines = [
        f"{_sha256_path(path)}  {path.relative_to(bundle.root).as_posix()}"
        for path in sorted(bundle.script_dir.glob("*.sbatch"))
    ]
    bundle.script_checksums_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return bundle.script_checksums_path


def manifest_sha256(path: str | Path) -> str:
    """Return the stable SHA-256 identity of a submission manifest."""
    return _sha256_path(Path(path))


def capture_environment_contract() -> dict[str, Any]:
    """Capture dependency and module identities that must not silently drift in a queued wave."""
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "datasets", "deepspeed", "vllm"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    torch_cuda: str | None = None
    try:
        import torch

        torch_cuda = str(torch.version.cuda)
    except ImportError:
        pass
    contract: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
        "packages": packages,
        "torch_cuda": torch_cuda,
    }
    return contract


def verify_submission_bundle(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    config_path: str | Path,
) -> list[str]:
    """Return integrity failures for the snapshot, config, manifest, or runtime environment."""
    manifest_file = Path(manifest_path)
    problems: list[str] = []
    if not manifest_file.is_file():
        return [f"submission manifest is missing: {manifest_file}"]
    actual_manifest_sha256 = manifest_sha256(manifest_file)
    if actual_manifest_sha256 != expected_manifest_sha256:
        problems.append("submission manifest SHA-256 does not match the generated Slurm script")
    try:
        manifest = _load_manifest(manifest_file)
    except ValueError as exc:
        return [str(exc)]

    repo_dir = manifest_file.parent / str(manifest["repository"]["snapshot_directory"])
    if not repo_dir.is_dir():
        problems.append(f"submission snapshot is missing: {repo_dir}")
    else:
        expected_commit = str(manifest["repository"]["git_commit"])
        try:
            observed_commit = _git(repo_dir, "rev-parse", "HEAD")
            if observed_commit != expected_commit:
                problems.append("submission snapshot commit does not match the manifest")
            if _git(repo_dir, "status", "--porcelain"):
                problems.append("submission snapshot is dirty")
        except RuntimeError as exc:
            problems.append(str(exc))

    config_file = Path(config_path)
    experiment_id = config_file.stem
    expected_config = manifest["configs"].get(experiment_id)
    if not isinstance(expected_config, dict):
        problems.append(f"config is not listed in the submission manifest: {config_file.name}")
    elif not config_file.is_file() or _sha256_path(config_file) != expected_config.get("file_sha256"):
        problems.append(f"resolved config SHA-256 does not match the submission manifest: {config_file.name}")

    expected_contract = manifest.get("environment_contract")
    if isinstance(expected_contract, dict):
        observed_contract = capture_environment_contract()
        if observed_contract != expected_contract:
            problems.append("runtime environment does not match the submission environment contract")
    return problems


def validate_result_bundle_provenance(record: dict[str, Any], result_path: str | Path) -> list[str]:
    """Validate an evidence record against the manifest staged beside bundle results."""
    environment = record.get("environment") if isinstance(record.get("environment"), dict) else {}
    bundle_id = environment.get("submission_bundle_id")
    if bundle_id is None:
        return []
    manifest_file = _find_result_manifest(Path(result_path))
    if manifest_file is None:
        return ["submission-bundle result is missing adjacent SUBMISSION_MANIFEST.json"]
    try:
        manifest = _load_manifest(manifest_file)
    except ValueError as exc:
        return [str(exc)]
    problems: list[str] = []
    if bundle_id != manifest["bundle_id"]:
        problems.append("submission_bundle_id does not match SUBMISSION_MANIFEST.json")
    if environment.get("submission_bundle_manifest_sha256") != manifest_sha256(manifest_file):
        problems.append("submission_bundle_manifest_sha256 does not match SUBMISSION_MANIFEST.json")
    experiment_id = str(record.get("experiment_id", ""))
    expected_config = manifest["configs"].get(experiment_id)
    if not isinstance(expected_config, dict):
        problems.append("experiment is not listed in SUBMISSION_MANIFEST.json")
    elif record.get("config_sha256") != expected_config.get("config_sha256"):
        problems.append("result config_sha256 does not match SUBMISSION_MANIFEST.json")
    if record.get("git_commit") != manifest["repository"]["git_commit"]:
        problems.append("result git_commit does not match SUBMISSION_MANIFEST.json")
    if record.get("platform_id") != manifest["platform_id"]:
        problems.append("result platform_id does not match SUBMISSION_MANIFEST.json")
    return problems


def _find_result_manifest(result_path: Path) -> Path | None:
    for parent in (result_path.parent, *result_path.parents):
        candidate = parent / "SUBMISSION_MANIFEST.json"
        if candidate.is_file():
            return candidate
    return None


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid submission manifest: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"submission manifest must be a JSON object: {path}")
    required = {"bundle_id", "platform_id", "wave", "repository", "configs"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"submission manifest is missing field(s): {', '.join(missing)}")
    repository = data.get("repository")
    if not isinstance(repository, dict) or not repository.get("git_commit") or not repository.get("snapshot_directory"):
        raise ValueError("submission manifest repository metadata is invalid")
    if not isinstance(data.get("configs"), dict):
        raise ValueError("submission manifest configs must be an object")
    return data


def _validate_execution_inputs_clean(repo_dir: Path) -> None:
    changed = set(_git_lines(repo_dir, "diff", "--name-only", "HEAD"))
    changed.update(_git_lines(repo_dir, "diff", "--cached", "--name-only"))
    changed.update(_git_lines(repo_dir, "ls-files", "--others", "--exclude-standard"))
    blockers = sorted(path for path in changed if _is_execution_input(path))
    if blockers:
        rendered = ", ".join(blockers[:8])
        suffix = " ..." if len(blockers) > 8 else ""
        raise ValueError(f"Cannot create submission bundle with uncommitted execution inputs: {rendered}{suffix}")


def _is_execution_input(path: str) -> bool:
    if path in _EXECUTION_INPUT_FILES:
        return True
    if path.startswith(_MUTABLE_RESULT_PATHS):
        return False
    return path.startswith(_EXECUTION_INPUT_PATHS)


def _git_repository_root(repository: str | Path) -> Path:
    return Path(_git(Path(repository), "rev-parse", "--show-toplevel")).resolve()


def _git(cwd: Path, *args: str) -> str:
    completed = _run_git(cwd, *args)
    return completed.stdout.strip()


def _git_lines(cwd: Path, *args: str) -> list[str]:
    value = _git(cwd, *args)
    return [line for line in value.splitlines() if line]


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise RuntimeError(f"git {' '.join(args)} failed: {message}")
    return completed


def _remove_new_bundle(repository: Path, root: Path) -> None:
    repo_dir = root / "repo"
    if repo_dir.exists():
        with suppress(RuntimeError):
            _run_git(repository, "worktree", "remove", "--force", str(repo_dir))
    shutil.rmtree(root, ignore_errors=True)


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
