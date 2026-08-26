from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from hpc_ai_perf.config import ExperimentConfig

ENGINE_MANIFEST_FILENAME = "hpc_ai_perf_engine_manifest.json"
ENGINE_MANIFEST_SCHEMA_VERSION = 1
ENGINE_API_CONTRACT = "legacy_llm_engine_v1"
_UNRESOLVED_ENVIRONMENT_VARIABLE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})")


def validate_tensorrt_llm_environment(
    config: ExperimentConfig,
    *,
    torch: Any,
    tensorrt_llm: Any,
) -> dict[str, int | float | str | bool | None]:
    """Build or load a TensorRT-LLM engine, then serve measured requests.

    E007 deliberately validates the saved-engine API used by the original
    experiment matrix. TensorRT-LLM releases that only expose the newer
    PyTorch backend must fail this check clearly because they do not satisfy
    the matrix's ``engine_build`` evidence requirement.
    """
    parameters = config.workload.parameters
    engine_mode = str(parameters["engine_mode"])
    engine_path = _resolve_path(str(parameters["engine_path"]), "engine_path")
    manifest_path = _manifest_path(parameters, engine_path)
    trtllm_version = str(getattr(tensorrt_llm, "__version__", "unknown"))
    expected_version = str(parameters["expected_tensorrt_llm_version"])
    if trtllm_version != expected_version:
        raise RuntimeError(
            f"E007 requires TensorRT-LLM {expected_version}, observed {trtllm_version}; "
            "the saved-engine API is version-sensitive"
        )
    _validate_requested_gpu(config, torch)
    build_seconds = 0.0

    if engine_mode == "build_and_validate":
        model_path = _resolve_path(str(parameters["model_path"]), "model_path")
        if not model_path.exists():
            raise RuntimeError(f"TensorRT-LLM model_path does not exist: {model_path}")
        _validate_model_revision(model_path, str(parameters["hf_model_revision"]))
        _require_empty_engine_directory(engine_path)
        started = time.perf_counter()
        build_runtime = _create_llm(tensorrt_llm, model_path, config, configure_build=True)
        save_engine = getattr(build_runtime, "save", None)
        if not callable(save_engine):
            _close_llm(build_runtime, torch)
            raise RuntimeError(
                "Installed TensorRT-LLM does not expose LLM.save(); E007 requires the legacy saved-engine "
                "contract. Use a compatible pinned release or revise the experiment to validate the PyTorch backend."
            )
        engine_path.mkdir(parents=True, exist_ok=True)
        try:
            save_engine(str(engine_path))
            build_seconds = time.perf_counter() - started
        finally:
            _close_llm(build_runtime, torch)
        engine_files = _engine_file_hashes(engine_path)
        if not engine_files:
            raise RuntimeError(f"TensorRT-LLM saved no engine files under {engine_path}")
        manifest = _build_manifest(config, torch, trtllm_version, model_path, engine_path, engine_files)
        _write_manifest(manifest_path, manifest)
    elif engine_mode == "validate_prebuilt":
        if not engine_path.is_dir():
            raise RuntimeError(f"TensorRT-LLM engine_path does not exist: {engine_path}")
    else:
        raise RuntimeError(f"Unsupported TensorRT-LLM engine_mode: {engine_mode}")

    manifest = _load_and_validate_manifest(manifest_path, config, torch, trtllm_version, engine_path)
    engine_runtime = _create_llm(tensorrt_llm, engine_path, config, configure_build=False)
    try:
        request_metrics = _serve_requests(engine_runtime, tensorrt_llm, torch, config)
    finally:
        _close_llm(engine_runtime, torch)

    return {
        "tensorrt_llm_available": True,
        "tensorrt_llm_import": True,
        "tensorrt_llm_version": trtllm_version,
        "engine_api_contract": ENGINE_API_CONTRACT,
        "engine_mode": engine_mode,
        "engine_build": True,
        "engine_built_this_run": engine_mode == "build_and_validate",
        "engine_build_seconds": build_seconds,
        "engine_loaded": True,
        "engine_manifest_valid": True,
        "engine_path": str(engine_path),
        "engine_manifest_path": str(manifest_path),
        "engine_manifest_sha256": _sha256_file(manifest_path),
        "engine_file_count": len(manifest["engine_files"]),
        "hf_model_name": str(parameters["hf_model_name"]),
        "hf_model_revision": str(parameters["hf_model_revision"]),
        "gpu_name": str(manifest["gpu_name"]),
        "gpu_compute_capability": str(manifest["gpu_compute_capability"]),
        "precision_observed": str(manifest["precision"]),
        **request_metrics,
    }


def _create_llm(
    tensorrt_llm: Any,
    model_path: Path,
    config: ExperimentConfig,
    *,
    configure_build: bool,
) -> Any:
    llm_class = getattr(tensorrt_llm, "LLM", None)
    if llm_class is None:
        raise RuntimeError("Installed TensorRT-LLM does not expose the high-level LLM API")
    parameters = config.workload.parameters
    kwargs: dict[str, Any] = {"model": str(model_path)}
    if configure_build:
        kwargs["tensor_parallel_size"] = int(parameters.get("tensor_parallel_size", 1))
        build_config_class = _optional_build_config(tensorrt_llm)
        if build_config_class is not None:
            build_config = build_config_class()
            _set_supported_attribute(build_config, "max_batch_size", int(parameters.get("max_batch_size", 1)))
            _set_supported_attribute(
                build_config,
                "max_num_tokens",
                int(parameters.get("max_input_tokens", 64)) + int(parameters.get("max_new_tokens", 8)),
            )
            kwargs["build_config"] = build_config
        else:
            _add_supported_keyword(llm_class, kwargs, "max_batch_size", int(parameters.get("max_batch_size", 1)))
            _add_supported_keyword(
                llm_class,
                kwargs,
                "max_seq_len",
                int(parameters.get("max_input_tokens", 64)) + int(parameters.get("max_new_tokens", 8)),
            )
    try:
        return llm_class(**kwargs)
    except TypeError as exc:
        raise RuntimeError(f"Installed TensorRT-LLM LLM API is incompatible with E007: {exc}") from exc


def _optional_build_config(tensorrt_llm: Any) -> Any | None:
    llmapi = getattr(tensorrt_llm, "llmapi", None)
    return getattr(llmapi, "BuildConfig", None) if llmapi is not None else None


def _set_supported_attribute(target: Any, name: str, value: Any) -> None:
    if hasattr(target, name):
        setattr(target, name, value)


def _add_supported_keyword(callable_object: Any, kwargs: dict[str, Any], name: str, value: Any) -> None:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return
    if name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    ):
        kwargs[name] = value


def _serve_requests(
    engine: Any,
    tensorrt_llm: Any,
    torch: Any,
    config: ExperimentConfig,
) -> dict[str, int | float | str | bool | None]:
    parameters = config.workload.parameters
    sampling_class = getattr(tensorrt_llm, "SamplingParams", None)
    if sampling_class is None:
        raise RuntimeError("Installed TensorRT-LLM does not expose SamplingParams")
    sampling = sampling_class(temperature=0.0, max_tokens=int(parameters.get("max_new_tokens", 8)))
    prompt = str(parameters.get("prompt", "The capital of France is"))
    warmup_requests = int(parameters.get("warmup_requests", 1))
    measured_requests = int(parameters.get("requests", 1))
    minimum_tokens = int(parameters.get("min_generated_tokens", 1))

    for _ in range(warmup_requests):
        _generate(engine, prompt, sampling)
    torch.cuda.synchronize()

    latencies: list[float] = []
    generated_tokens = 0
    served_requests = 0
    for _ in range(measured_requests):
        started = time.perf_counter()
        outputs = _generate(engine, prompt, sampling)
        torch.cuda.synchronize()
        latencies.append(max(time.perf_counter() - started, 1e-12))
        output_count, token_count = _count_outputs(outputs)
        served_requests += output_count
        generated_tokens += token_count

    if served_requests < measured_requests:
        raise RuntimeError(f"TensorRT-LLM served {served_requests} outputs for {measured_requests} measured requests")
    if generated_tokens < minimum_tokens:
        raise RuntimeError(f"TensorRT-LLM generated {generated_tokens} tokens; expected at least {minimum_tokens}")
    measured_seconds = sum(latencies)
    return {
        "request_served": True,
        "served_requests": served_requests,
        "generated_tokens": generated_tokens,
        "total_tokens": generated_tokens,
        "throughput_unit": "generated_tokens_per_second",
        "throughput_tokens_sec": generated_tokens / max(measured_seconds, 1e-12),
        "measured_seconds": measured_seconds,
        "gpu_seconds": measured_seconds,
        "gpu_hours": measured_seconds / 3600,
        "latency_p50_ms": _percentile(latencies, 0.50) * 1000,
        "latency_p95_ms": _percentile(latencies, 0.95) * 1000,
        "warmup_iterations": warmup_requests,
        "measured_iterations": measured_requests,
        "framework_observed": "tensorrt_llm",
    }


def _generate(engine: Any, prompt: str, sampling: Any) -> Any:
    try:
        return engine.generate([prompt], sampling_params=sampling, use_tqdm=False)
    except TypeError:
        return engine.generate([prompt], sampling)


def _count_outputs(outputs: Any) -> tuple[int, int]:
    if outputs is None:
        return 0, 0
    output_list = outputs if isinstance(outputs, list | tuple) else [outputs]
    generated_tokens = 0
    for output in output_list:
        candidates = getattr(output, "outputs", [])
        for candidate in candidates:
            token_ids = getattr(candidate, "token_ids", None)
            if token_ids is not None:
                generated_tokens += len(token_ids)
                continue
            text = str(getattr(candidate, "text", ""))
            generated_tokens += len(text.split())
    return len(output_list), generated_tokens


def _build_manifest(
    config: ExperimentConfig,
    torch: Any,
    trtllm_version: str,
    model_path: Path,
    engine_path: Path,
    engine_files: dict[str, str],
) -> dict[str, Any]:
    parameters = config.workload.parameters
    capability = torch.cuda.get_device_capability(0)
    return {
        "schema_version": ENGINE_MANIFEST_SCHEMA_VERSION,
        "engine_api_contract": ENGINE_API_CONTRACT,
        "experiment_id": config.experiment_id,
        "model_name": str(parameters["hf_model_name"]),
        "model_revision": str(parameters["hf_model_revision"]),
        "model_path": str(model_path.resolve()),
        "precision": config.runtime.precision,
        "tensor_parallel_size": int(parameters.get("tensor_parallel_size", 1)),
        "max_batch_size": int(parameters.get("max_batch_size", 1)),
        "max_input_tokens": int(parameters.get("max_input_tokens", 64)),
        "max_new_tokens": int(parameters.get("max_new_tokens", 8)),
        "tensorrt_llm_version": trtllm_version,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "unknown")),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_compute_capability": f"{int(capability[0])}.{int(capability[1])}",
        "engine_path": str(engine_path),
        "engine_files": engine_files,
    }


def _load_and_validate_manifest(
    manifest_path: Path,
    config: ExperimentConfig,
    torch: Any,
    trtllm_version: str,
    engine_path: Path,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise RuntimeError(f"TensorRT-LLM engine manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read TensorRT-LLM engine manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("TensorRT-LLM engine manifest must be a JSON object")

    parameters = config.workload.parameters
    capability = torch.cuda.get_device_capability(0)
    expected = {
        "schema_version": ENGINE_MANIFEST_SCHEMA_VERSION,
        "engine_api_contract": ENGINE_API_CONTRACT,
        "model_name": str(parameters["hf_model_name"]),
        "model_revision": str(parameters["hf_model_revision"]),
        "precision": config.runtime.precision,
        "tensor_parallel_size": int(parameters.get("tensor_parallel_size", 1)),
        "tensorrt_llm_version": trtllm_version,
        "gpu_compute_capability": f"{int(capability[0])}.{int(capability[1])}",
    }
    mismatches = [
        f"{key}={manifest.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("TensorRT-LLM engine manifest is incompatible: " + "; ".join(mismatches))
    recorded_files = manifest.get("engine_files")
    if not isinstance(recorded_files, dict) or not recorded_files:
        raise RuntimeError("TensorRT-LLM engine manifest has no engine file hashes")
    observed_files = _engine_file_hashes(engine_path)
    if observed_files != recorded_files:
        raise RuntimeError("TensorRT-LLM engine files do not match the manifest hashes")
    return manifest


def _manifest_path(parameters: dict[str, Any], engine_path: Path) -> Path:
    configured = parameters.get("engine_manifest_path")
    if configured:
        return _resolve_path(str(configured), "engine_manifest_path")
    return engine_path / ENGINE_MANIFEST_FILENAME


def _resolve_path(value: str, field_name: str) -> Path:
    expanded = os.path.expandvars(value)
    if _UNRESOLVED_ENVIRONMENT_VARIABLE.search(expanded):
        raise RuntimeError(f"TensorRT-LLM {field_name} contains an unresolved environment variable: {value}")
    return Path(expanded)


def _validate_model_revision(model_path: Path, expected_revision: str) -> None:
    resolved_path = str(model_path.resolve())
    if expected_revision in resolved_path:
        return
    marker_path = model_path / "hpc_ai_perf_model_revision.txt"
    if marker_path.is_file() and marker_path.read_text(encoding="utf-8").strip() == expected_revision:
        return
    raise RuntimeError(
        "TensorRT-LLM model_path does not prove the configured Hugging Face revision; "
        f"use a snapshot path containing {expected_revision} or write that revision to {marker_path}"
    )


def _validate_requested_gpu(config: ExperimentConfig, torch: Any) -> None:
    expected_count = int(config.workload.parameters.get("expected_gpus", config.hardware.gpus))
    observed_count = int(torch.cuda.device_count())
    if observed_count != expected_count:
        raise RuntimeError(f"E007 expected {expected_count} visible GPUs, observed {observed_count}")
    requested_model = str(config.hardware.gpu_model or "").lower()
    observed_model = str(torch.cuda.get_device_name(0)).lower()
    if "h100" in requested_model and "h100" not in observed_model:
        raise RuntimeError(f"E007 requested {config.hardware.gpu_model}, observed {torch.cuda.get_device_name(0)}")
    if "l40s" in requested_model and "l40s" not in observed_model:
        raise RuntimeError(f"E007 requested {config.hardware.gpu_model}, observed {torch.cuda.get_device_name(0)}")


def _require_empty_engine_directory(engine_path: Path) -> None:
    if engine_path.exists() and not engine_path.is_dir():
        raise RuntimeError(f"TensorRT-LLM engine_path is not a directory: {engine_path}")
    if engine_path.is_dir() and any(engine_path.iterdir()):
        raise RuntimeError(
            f"TensorRT-LLM build output is not empty: {engine_path}; use a unique path for each validation run"
        )


def _engine_file_hashes(engine_path: Path) -> dict[str, str]:
    if not engine_path.is_dir():
        return {}
    files = [path for path in engine_path.rglob("*") if path.is_file() and path.name != ENGINE_MANIFEST_FILENAME]
    return {str(path.relative_to(engine_path)): _sha256_file(path) for path in sorted(files)}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def _close_llm(engine: Any, torch: Any) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del engine
    gc.collect()
    empty_cache = getattr(torch.cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
