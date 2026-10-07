"""Multi-replica vLLM routing for explicit data-parallel serving layouts."""

from __future__ import annotations

import multiprocessing as mp
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutedMetrics:
    arrival_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None
    first_scheduled_time: float | None = None


@dataclass(frozen=True)
class RoutedCandidate:
    token_ids: list[int]


@dataclass(frozen=True)
class RoutedOutput:
    outputs: list[RoutedCandidate]
    metrics: RoutedMetrics | None


def build_gpu_groups(
    *,
    gpu_count: int,
    tensor_parallel_size: int,
    replica_count: int,
    visible_devices: str | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Partition visible GPU identifiers into isolated tensor-parallel groups."""
    if gpu_count < 1 or tensor_parallel_size < 1 or replica_count < 1:
        raise ValueError("gpu_count, tensor_parallel_size, and replica_count must be positive")
    expected = tensor_parallel_size * replica_count
    if gpu_count != expected:
        raise ValueError(f"replica layout requires {expected} GPUs, observed {gpu_count}")
    devices = tuple(
        token.strip()
        for token in (visible_devices or ",".join(str(index) for index in range(gpu_count))).split(",")
        if token.strip()
    )
    if len(devices) != gpu_count:
        raise ValueError(f"CUDA_VISIBLE_DEVICES exposes {len(devices)} devices, expected {gpu_count}")
    return tuple(
        tuple(devices[offset : offset + tensor_parallel_size]) for offset in range(0, gpu_count, tensor_parallel_size)
    )


def route_prompt_indices(
    prompt_count: int,
    replica_count: int,
    *,
    request_routing: str = "round_robin",
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic prompt assignments for the configured replica policy."""
    if prompt_count < 0 or replica_count < 1:
        raise ValueError("prompt_count must be non-negative and replica_count must be positive")
    if request_routing not in {"round_robin", "least_loaded"}:
        raise ValueError(f"unsupported request_routing={request_routing!r}")
    assignments = [[] for _ in range(replica_count)]
    for prompt_index in range(prompt_count):
        if request_routing == "round_robin":
            replica_index = prompt_index % replica_count
        else:
            lowest_load = min(len(items) for items in assignments)
            replica_index = next(index for index, items in enumerate(assignments) if len(items) == lowest_load)
        assignments[replica_index].append(prompt_index)
    return tuple(tuple(items) for items in assignments)


class VLLMReplicaRouter:
    """Own one vLLM engine process per replica and route request batches to them."""

    def __init__(
        self,
        *,
        model_name: str,
        model_revision: str,
        dtype: str,
        tensor_parallel_size: int,
        gpu_groups: tuple[tuple[str, ...], ...],
        request_routing: str = "round_robin",
        llm_kwargs: dict[str, Any] | None = None,
        startup_timeout_seconds: float = 900.0,
    ) -> None:
        if not gpu_groups:
            raise ValueError("at least one replica GPU group is required")
        if request_routing not in {"round_robin", "least_loaded"}:
            raise ValueError(f"unsupported request_routing={request_routing!r}")
        self._model_name = model_name
        self._model_revision = model_revision
        self._dtype = dtype
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_groups = gpu_groups
        self._request_routing = request_routing
        self._llm_kwargs = dict(llm_kwargs or {})
        self._startup_timeout_seconds = startup_timeout_seconds
        self._processes: list[mp.Process] = []
        self._connections: list[Any] = []
        self.vocab_size = 0
        self.observed_quantization: str | None = None
        self.observed_dtype = dtype

    def __enter__(self) -> VLLMReplicaRouter:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._processes:
            raise RuntimeError("vLLM replica router is already started")
        context = mp.get_context("spawn")
        try:
            for gpu_group in self._gpu_groups:
                parent, child = context.Pipe(duplex=True)
                process = context.Process(
                    target=_vllm_replica_worker,
                    args=(
                        child,
                        gpu_group,
                        self._model_name,
                        self._model_revision,
                        self._dtype,
                        self._tensor_parallel_size,
                        self._llm_kwargs,
                    ),
                    daemon=True,
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)

            for connection in self._connections:
                if not connection.poll(self._startup_timeout_seconds):
                    raise TimeoutError("timed out waiting for a vLLM replica to initialize")
                message = connection.recv()
                if message.get("kind") != "ready":
                    raise RuntimeError(message.get("error", "vLLM replica failed during initialization"))
                self.vocab_size = max(self.vocab_size, int(message["vocab_size"]))
                self.observed_quantization = message.get("quantization")
                self.observed_dtype = str(message.get("dtype", self.observed_dtype))
        except Exception:
            self.close()
            raise

    def generate(self, prompts: list[dict[str, list[int]]], *, max_new_tokens: int) -> list[RoutedOutput]:
        if not self._connections:
            raise RuntimeError("vLLM replica router is not started")
        assignments = route_prompt_indices(
            len(prompts),
            len(self._connections),
            request_routing=self._request_routing,
        )
        outputs: list[RoutedOutput | None] = [None] * len(prompts)
        for connection, prompt_indices in zip(self._connections, assignments, strict=True):
            if prompt_indices:
                connection.send(
                    {
                        "kind": "generate",
                        "prompts": [prompts[index] for index in prompt_indices],
                        "max_new_tokens": max_new_tokens,
                    }
                )
        for connection, prompt_indices in zip(self._connections, assignments, strict=True):
            if not prompt_indices:
                continue
            message = connection.recv()
            if message.get("kind") != "result":
                raise RuntimeError(message.get("error", "vLLM replica failed during generation"))
            for prompt_index, output in zip(prompt_indices, message["outputs"], strict=True):
                outputs[prompt_index] = _deserialize_output(output)
        return [output for output in outputs if output is not None]

    def close(self) -> None:
        for connection in self._connections:
            with suppress(BrokenPipeError, EOFError, OSError):
                connection.send({"kind": "shutdown"})
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        self._processes.clear()


def _vllm_replica_worker(
    connection: Any,
    gpu_group: tuple[str, ...],
    model_name: str,
    model_revision: str,
    dtype: str,
    tensor_parallel_size: int,
    llm_kwargs: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_group)
    try:
        from vllm import LLM, SamplingParams

        engine_kwargs = {
            "model": model_name,
            "revision": model_revision,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": True,
            **llm_kwargs,
        }
        llm = LLM(**engine_kwargs)
        model_config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
        connection.send(
            {
                "kind": "ready",
                "vocab_size": int(llm.get_tokenizer().vocab_size),
                "quantization": getattr(model_config, "quantization", None),
                "dtype": str(getattr(model_config, "dtype", dtype)),
            }
        )
        while True:
            message = connection.recv()
            if message.get("kind") == "shutdown":
                return
            if message.get("kind") != "generate":
                raise RuntimeError(f"unknown replica command: {message.get('kind')}")
            sampling = SamplingParams(temperature=0.0, max_tokens=int(message["max_new_tokens"]))
            outputs = llm.generate(message["prompts"], sampling_params=sampling, use_tqdm=False)
            connection.send({"kind": "result", "outputs": [_serialize_output(output) for output in outputs]})
    except Exception as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})


def _serialize_output(output: Any) -> dict[str, Any]:
    request_metrics = getattr(output, "metrics", None)
    return {
        "token_count": sum(len(candidate.token_ids) for candidate in output.outputs),
        "metrics": {
            "arrival_time": getattr(request_metrics, "arrival_time", None),
            "first_token_time": getattr(request_metrics, "first_token_time", None),
            "finished_time": getattr(request_metrics, "finished_time", None),
            "first_scheduled_time": getattr(request_metrics, "first_scheduled_time", None),
        }
        if request_metrics is not None
        else None,
    }


def _deserialize_output(payload: dict[str, Any]) -> RoutedOutput:
    token_count = int(payload.get("token_count", 0))
    metric_payload = payload.get("metrics")
    metrics = RoutedMetrics(**metric_payload) if metric_payload is not None else None
    return RoutedOutput(outputs=[RoutedCandidate(token_ids=[0] * token_count)], metrics=metrics)
