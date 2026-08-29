from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeCheck:
    """One human-readable runtime preflight check."""

    name: str
    status: str
    detail: str
    remediation: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == "fail"

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "remediation": self.remediation,
        }


def prepare_deepspeed_import_environment(torch: Any | None = None) -> dict[str, str | bool | None]:
    """Normalize CUDA discovery before importing DeepSpeed.

    Some cluster module stacks expose a CUDA runtime through PyTorch but leave
    ``CUDA_HOME`` pointing at a toolkit path without ``bin/nvcc``. DeepSpeed's
    optional-op compatibility probes can try to execute that missing compiler
    at import time, even for experiments that only need the ZeRO runtime. This
    helper either points CUDA_HOME at a real nvcc, or disables the DeepSpeed
    CUDA-op detection probe while leaving the actual CUDA runtime check to the
    smoke test itself.
    """

    cpp_extension = _torch_cpp_extension(torch)
    cuda_home_before = _cuda_home_from_torch_or_env(cpp_extension)
    nvcc_path_before = shutil.which("nvcc")
    cuda_home_nvcc_before = _nvcc_under(cuda_home_before)
    selected_cuda_home = cuda_home_before
    ignored_detection = False

    if not cuda_home_nvcc_before:
        if nvcc_path_before:
            selected_cuda_home = str(Path(nvcc_path_before).resolve().parent.parent)
            os.environ["CUDA_HOME"] = selected_cuda_home
            _set_torch_cuda_home(cpp_extension, selected_cuda_home)
        else:
            selected_cuda_home = None
            os.environ.pop("CUDA_HOME", None)
            os.environ.pop("CUDA_PATH", None)
            _set_torch_cuda_home(cpp_extension, None)
            os.environ.setdefault("DS_IGNORE_CUDA_DETECTION", "1")
            ignored_detection = True

    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    cuda_home_after = _cuda_home_from_torch_or_env(cpp_extension)
    nvcc_path_after = shutil.which("nvcc")
    cuda_home_nvcc_after = _nvcc_under(cuda_home_after)
    return {
        "cuda_home_before": cuda_home_before,
        "cuda_home_after": cuda_home_after,
        "nvcc_path_before": nvcc_path_before,
        "nvcc_path_after": nvcc_path_after,
        "cuda_home_nvcc_before": str(cuda_home_nvcc_before) if cuda_home_nvcc_before else None,
        "cuda_home_nvcc_after": str(cuda_home_nvcc_after) if cuda_home_nvcc_after else None,
        "nvcc_available": bool(cuda_home_nvcc_after or nvcc_path_after),
        "deepspeed_cuda_detection_ignored": os.environ.get("DS_IGNORE_CUDA_DETECTION") == "1" or ignored_detection,
        "pytorch_nvml_based_cuda_check": os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK"),
        "selected_cuda_home": selected_cuda_home,
    }


def run_cluster_runtime_checks(
    *,
    expected_gpu_model: str | None = None,
    expected_gpus: int | None = None,
    require_deepspeed: bool = False,
    require_vllm: bool = False,
    require_clean_git: bool = True,
) -> tuple[RuntimeCheck, ...]:
    """Check active node runtime before submitting or accepting GPU evidence."""

    checks: list[RuntimeCheck] = []
    if require_clean_git:
        checks.append(_git_clean_check())

    torch_module = None
    try:
        import torch

        torch_module = torch
    except Exception as exc:  # pragma: no cover - depends on active environment
        checks.append(
            RuntimeCheck(
                "torch_import",
                "fail",
                f"PyTorch import failed: {type(exc).__name__}: {exc}",
                "Load the cluster PyTorch module and activate the project virtual environment.",
            )
        )

    if torch_module is None:
        return tuple(checks)

    checks.extend(_torch_cuda_checks(torch_module, expected_gpu_model, expected_gpus))

    if require_deepspeed:
        checks.extend(_deepspeed_checks(torch_module))
    if require_vllm:
        checks.extend(_vllm_checks(torch_module))
    return tuple(checks)


def format_runtime_checks(checks: tuple[RuntimeCheck, ...]) -> str:
    """Render runtime checks as a compact terminal report."""

    lines: list[str] = []
    for check in checks:
        prefix = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(check.status, check.status.upper())
        lines.append(f"[{prefix}] {check.name}: {check.detail}")
        if check.remediation:
            lines.append(f"       fix: {check.remediation}")
    return "\n".join(lines) + ("\n" if lines else "")


def runtime_checks_to_json(checks: tuple[RuntimeCheck, ...]) -> str:
    """Serialize runtime checks for automation."""

    return json.dumps([check.to_dict() for check in checks], indent=2) + "\n"


def _git_clean_check() -> RuntimeCheck:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RuntimeCheck(
            "git_clean",
            "fail",
            f"Could not inspect git status: {type(exc).__name__}: {exc}",
            "Run from a valid repository checkout before launching matrix jobs.",
        )
    dirty_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if dirty_lines:
        preview = "; ".join(dirty_lines[:5])
        return RuntimeCheck(
            "git_clean",
            "fail",
            f"working tree is dirty ({len(dirty_lines)} path(s)): {preview}",
            "Commit, stash, or remove generated files before submitting jobs.",
        )
    return RuntimeCheck("git_clean", "pass", "working tree is clean")


def _torch_cuda_checks(
    torch: Any,
    expected_gpu_model: str | None,
    expected_gpus: int | None,
) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = [
        RuntimeCheck(
            "torch_import",
            "pass",
            f"torch {getattr(torch, '__version__', 'unknown')} cuda {getattr(torch.version, 'cuda', 'unknown')}",
        )
    ]
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on active environment
        checks.append(
            RuntimeCheck(
                "cuda_available",
                "fail",
                f"CUDA availability check failed: {type(exc).__name__}: {exc}",
                "Run inside a Slurm GPU allocation and load the CUDA/PyTorch module stack.",
            )
        )
        return checks

    if not cuda_available:
        checks.append(
            RuntimeCheck(
                "cuda_available",
                "fail",
                "torch.cuda.is_available() is false",
                "Run this preflight inside an allocated GPU node, not on the login node.",
            )
        )
        return checks

    count = int(torch.cuda.device_count())
    checks.append(RuntimeCheck("cuda_available", "pass", f"{count} CUDA device(s) visible"))
    if expected_gpus is not None:
        status = "pass" if count == expected_gpus else "fail"
        checks.append(
            RuntimeCheck(
                "cuda_device_count",
                status,
                f"expected {expected_gpus}, observed {count}",
                "Request the same GPU count as the wave row before launching the check." if status == "fail" else None,
            )
        )

    names = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    checks.append(RuntimeCheck("cuda_devices", "pass", ", ".join(names)))
    if expected_gpu_model:
        token = expected_gpu_model.lower()
        mismatches = [name for name in names if token not in name.lower()]
        status = "fail" if mismatches else "pass"
        checks.append(
            RuntimeCheck(
                "cuda_device_model",
                status,
                f"expected name containing {expected_gpu_model!r}; observed {', '.join(names)}",
                "Use the matching Slurm partition/profile for this platform." if status == "fail" else None,
            )
        )
    return checks


def _deepspeed_checks(torch: Any) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = []
    metadata = prepare_deepspeed_import_environment(torch)
    deepspeed_version = _distribution_version("deepspeed")
    if metadata["nvcc_available"]:
        checks.append(
            RuntimeCheck(
                "deepspeed_nvcc",
                "pass",
                f"nvcc available via {metadata['cuda_home_nvcc_after'] or metadata['nvcc_path_after']}",
            )
        )
    else:
        checks.append(
            RuntimeCheck(
                "deepspeed_nvcc",
                "warn",
                "no nvcc compiler found; DeepSpeed CUDA-op detection is disabled for ZeRO runtime smoke checks",
                "Load a full CUDA toolkit module if you need to build custom DeepSpeed CUDA ops.",
            )
        )
    if metadata["deepspeed_cuda_detection_ignored"]:
        checks.append(
            RuntimeCheck(
                "deepspeed_cuda_probe",
                "warn",
                "DS_IGNORE_CUDA_DETECTION=1 is active to avoid stale CUDA_HOME/nvcc probes",
            )
        )

    if importlib.util.find_spec("deepspeed") is None:
        checks.append(
            RuntimeCheck(
                "deepspeed_import",
                "fail",
                "DeepSpeed is not installed",
                "Install the distributed extra or the site-provided DeepSpeed package.",
            )
        )
        return checks
    if deepspeed_version and _version_less_than(deepspeed_version, (0, 18, 9)) and not metadata["nvcc_available"]:
        checks.append(
            RuntimeCheck(
                "deepspeed_version",
                "fail",
                f"deepspeed {deepspeed_version} is installed without an available nvcc compiler",
                "Upgrade to deepspeed>=0.18.9,<0.20, or load a full CUDA toolkit whose CUDA_HOME/bin/nvcc exists.",
            )
        )
        return checks
    try:
        import deepspeed
    except Exception as exc:  # pragma: no cover - depends on active environment
        checks.append(
            RuntimeCheck(
                "deepspeed_import",
                "fail",
                f"{type(exc).__name__}: {exc}",
                "Upgrade to deepspeed>=0.18.9,<0.20, or load a full CUDA toolkit whose CUDA_HOME/bin/nvcc exists.",
            )
        )
        return checks
    checks.append(RuntimeCheck("deepspeed_import", "pass", f"deepspeed {getattr(deepspeed, '__version__', 'unknown')}"))
    try:
        from deepspeed.accelerator import get_accelerator

        accelerator = str(get_accelerator().device_name())
    except Exception as exc:  # pragma: no cover - depends on active environment
        checks.append(
            RuntimeCheck(
                "deepspeed_accelerator",
                "fail",
                f"could not inspect accelerator: {type(exc).__name__}: {exc}",
            )
        )
    else:
        status = "pass" if "cuda" in accelerator.lower() else "fail"
        checks.append(
            RuntimeCheck(
                "deepspeed_accelerator",
                status,
                accelerator,
                "DeepSpeed must detect CUDA for E005 and later ZeRO rows." if status == "fail" else None,
            )
        )
    return checks


def _vllm_checks(torch: Any) -> list[RuntimeCheck]:
    if importlib.util.find_spec("vllm") is None:
        return [
            RuntimeCheck(
                "vllm_import",
                "fail",
                "vLLM is not installed",
                "Install or build vLLM for the active cluster CUDA/PyTorch runtime before E006.",
            )
        ]
    try:
        import vllm
    except Exception as exc:  # pragma: no cover - depends on active environment
        torch_cuda = getattr(getattr(torch, "version", None), "cuda", "unknown")
        message = f"{type(exc).__name__}: {exc}"
        remediation = "Install or build vLLM for the active cluster CUDA/PyTorch runtime before E006."
        if "libcudart.so.13" in str(exc) and str(torch_cuda).startswith("12."):
            remediation = (
                "Uninstall the CUDA-13 vLLM wheel, then install/build vLLM against the cluster CUDA 12.x/PyTorch stack."
            )
        return [
            RuntimeCheck(
                "vllm_import",
                "fail",
                f"{message}; torch CUDA {torch_cuda}",
                remediation,
            )
        ]
    return [RuntimeCheck("vllm_import", "pass", f"vllm {getattr(vllm, '__version__', 'unknown')}")]


def _torch_cpp_extension(torch: Any | None) -> Any | None:
    if torch is None:
        return None
    try:
        import torch.utils.cpp_extension as cpp_extension
    except Exception:
        return None
    return cpp_extension


def _cuda_home_from_torch_or_env(cpp_extension: Any | None) -> str | None:
    cuda_home = getattr(cpp_extension, "CUDA_HOME", None) if cpp_extension is not None else None
    if cuda_home:
        return str(cuda_home)
    return os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")


def _set_torch_cuda_home(cpp_extension: Any | None, value: str | None) -> None:
    if cpp_extension is not None and hasattr(cpp_extension, "CUDA_HOME"):
        cpp_extension.CUDA_HOME = value


def _nvcc_under(cuda_home: str | None) -> Path | None:
    if not cuda_home:
        return None
    candidate = Path(cuda_home) / "bin" / "nvcc"
    return candidate if candidate.exists() and os.access(candidate, os.X_OK) else None


def _distribution_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _version_less_than(version: str, minimum: tuple[int, int, int]) -> bool:
    parts = []
    for raw_part in version.split("+", 1)[0].split(".")[:3]:
        numeric = "".join(character for character in raw_part if character.isdigit())
        parts.append(int(numeric) if numeric else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3]) < minimum
