from __future__ import annotations

import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


@dataclass(frozen=True)
class TransformerWorkload(Workload):
    """Placeholder adapter for transformer workloads that need GPU-capable dependencies."""

    name: str
    display_name: str
    required_packages: tuple[str, ...] = ("torch", "transformers")

    def prepare(self, config: ExperimentConfig) -> None:
        missing = [package for package in self.required_packages if find_spec(package) is None]
        if missing:
            package_list = ", ".join(missing)
            raise RuntimeError(
                f"{self.display_name} requires optional dependencies: {package_list}. "
                "Install them with: python -m pip install -e '.[workloads]'"
            )

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        raise NotImplementedError(
            f"{self.display_name} adapter is scaffolded but execution is not implemented yet for {config.mode}."
        )


class GPT2SmallWorkload(TransformerWorkload):
    """Minimal Hugging Face/PyTorch GPT-2 workload for smoke runs and baseline plumbing."""

    def __init__(self) -> None:
        super().__init__(name="gpt2_small", display_name="GPT-2 Small")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode == "training":
            return self._run_training(config)
        if config.mode == "inference":
            return self._run_inference(config)
        raise NotImplementedError(f"GPT-2 Small does not support mode: {config.mode}")

    def _run_training(self, config: ExperimentConfig) -> WorkloadRunOutput:
        torch, auto_model, _ = _load_causal_lm_stack()
        parameters = config.workload.parameters
        device = _device(torch, parameters)
        dtype = _torch_dtype(torch, config.runtime.precision, device)
        model_name = str(parameters.get("hf_model_name", config.workload.model))
        steps = config.runtime.max_steps or int(parameters.get("steps", 1))
        batch_size = int(parameters.get("micro_batch_size", parameters.get("per_gpu_batch_size", 1)))
        sequence_length = int(parameters.get("sequence_length", 128))
        learning_rate = float(parameters.get("learning_rate", 5e-5))

        model = auto_model.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(batch_size, sequence_length),
            device=device,
        )

        start = time.perf_counter()
        final_loss = 0.0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        elapsed = max(time.perf_counter() - start, 1e-9)
        total_tokens = steps * batch_size * sequence_length

        return WorkloadRunOutput(
            metrics={
                "steps": steps,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "total_tokens": total_tokens,
                "loss": final_loss,
                "throughput_tokens_sec": total_tokens / elapsed,
                "device": str(device),
                "hf_model_name": model_name,
            }
        )

    def _run_inference(self, config: ExperimentConfig) -> WorkloadRunOutput:
        return _run_causal_lm_inference(config)


class DistilBERTWorkload(TransformerWorkload):
    """Minimal Hugging Face/PyTorch DistilBERT workload for encoder smoke runs."""

    def __init__(self) -> None:
        super().__init__(name="distilbert", display_name="DistilBERT")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode == "training":
            return self._run_training(config)
        if config.mode == "inference":
            return self._run_inference(config)
        raise NotImplementedError(f"DistilBERT does not support mode: {config.mode}")

    def _run_training(self, config: ExperimentConfig) -> WorkloadRunOutput:
        torch, auto_model, _ = _load_sequence_classification_stack()
        parameters = config.workload.parameters
        device = _device(torch, parameters)
        dtype = _torch_dtype(torch, config.runtime.precision, device)
        model_name = str(parameters.get("hf_model_name", config.workload.model))
        steps = config.runtime.max_steps or int(parameters.get("steps", 1))
        batch_size = int(parameters.get("micro_batch_size", parameters.get("per_gpu_batch_size", 1)))
        sequence_length = int(parameters.get("sequence_length", 128))
        num_labels = int(parameters.get("num_labels", 2))
        learning_rate = float(parameters.get("learning_rate", 5e-5))

        model = auto_model.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
        ).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(batch_size, sequence_length),
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(low=0, high=num_labels, size=(batch_size,), device=device)

        start = time.perf_counter()
        final_loss = 0.0
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        elapsed = max(time.perf_counter() - start, 1e-9)
        total_samples = steps * batch_size
        total_tokens = total_samples * sequence_length

        return WorkloadRunOutput(
            metrics={
                "steps": steps,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "total_samples": total_samples,
                "total_tokens": total_tokens,
                "loss": final_loss,
                "throughput_samples_sec": total_samples / elapsed,
                "throughput_tokens_sec": total_tokens / elapsed,
                "device": str(device),
                "hf_model_name": model_name,
            }
        )

    def _run_inference(self, config: ExperimentConfig) -> WorkloadRunOutput:
        torch, auto_model, auto_tokenizer = _load_sequence_classification_stack()
        parameters = config.workload.parameters
        device = _device(torch, parameters)
        dtype = _torch_dtype(torch, config.runtime.precision, device)
        model_name = str(parameters.get("hf_model_name", config.workload.model))
        text = str(parameters.get("text", "Transformer inference benchmark sentence."))
        num_labels = int(parameters.get("num_labels", 2))

        tokenizer = auto_tokenizer.from_pretrained(model_name)
        model = auto_model.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True,
        ).to(device)
        model.eval()
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = max(time.perf_counter() - start, 1e-9)
        predicted_label = int(outputs.logits.argmax(dim=-1).detach().cpu()[0])
        tokens = int(inputs["input_ids"].shape[-1])

        return WorkloadRunOutput(
            metrics={
                "samples": 1,
                "input_tokens": tokens,
                "predicted_label": predicted_label,
                "throughput_samples_sec": 1 / elapsed,
                "throughput_tokens_sec": tokens / elapsed,
                "latency_p50_ms": elapsed * 1000,
                "latency_p95_ms": elapsed * 1000,
                "device": str(device),
                "hf_model_name": model_name,
            }
        )


class TinyLlamaWorkload(TransformerWorkload):
    """Minimal TinyLlama-style causal LM inference workload."""

    def __init__(self) -> None:
        super().__init__(name="tinyllama", display_name="TinyLlama")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        if config.mode == "inference":
            return _run_causal_lm_inference(config)
        raise NotImplementedError("TinyLlama training is not implemented yet; use inference configs for this adapter.")


def transformer_workloads() -> list[TransformerWorkload]:
    return [
        GPT2SmallWorkload(),
        DistilBERTWorkload(),
        TinyLlamaWorkload(),
    ]


def _load_causal_lm_stack() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _load_sequence_classification_stack() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _run_causal_lm_inference(config: ExperimentConfig) -> WorkloadRunOutput:
    torch, auto_model, auto_tokenizer = _load_causal_lm_stack()
    parameters = config.workload.parameters
    device = _device(torch, parameters)
    dtype = _torch_dtype(torch, config.runtime.precision, device)
    model_name = str(parameters.get("hf_model_name", config.workload.model))
    prompt = str(parameters.get("prompt", "Benchmarking transformer inference on H100 systems."))
    max_new_tokens = int(parameters.get("max_new_tokens", 32))

    tokenizer = auto_tokenizer.from_pretrained(model_name)
    model = auto_model.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = max(time.perf_counter() - start, 1e-9)
    generated_tokens = int(outputs.shape[-1] - inputs["input_ids"].shape[-1])

    return WorkloadRunOutput(
        metrics={
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": generated_tokens,
            "total_tokens": generated_tokens,
            "throughput_tokens_sec": generated_tokens / elapsed,
            "latency_p50_ms": elapsed * 1000,
            "latency_p95_ms": elapsed * 1000,
            "device": str(device),
            "hf_model_name": model_name,
        }
    )


def _device(torch: Any, parameters: dict[str, Any] | None = None) -> Any:
    requested_device = str((parameters or {}).get("device", "auto")).lower()
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but CUDA is not available.")
    if requested_device == "mps" and not _mps_available(torch):
        raise RuntimeError("Requested MPS device, but Apple MPS is not available.")
    if requested_device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device setting: {requested_device}")
    return torch.device(requested_device)


def _mps_available(torch: Any) -> bool:
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def _torch_dtype(torch: Any, precision: str, device: Any) -> Any:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if precision == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    raise NotImplementedError(f"Transformer workload does not support precision yet: {precision}")
