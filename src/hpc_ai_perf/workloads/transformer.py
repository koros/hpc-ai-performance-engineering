from __future__ import annotations

import time
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


@dataclass(frozen=True)
class TransformerWorkload(Workload):
    """Base adapter for Hugging Face/PyTorch transformer workloads.

    Transformer workloads are optional because the synthetic smoke path should
    remain runnable on a minimal Python environment. This base class centralises
    dependency checks and a clear "not implemented" fallback so each concrete
    adapter can focus on the training or inference behavior it actually
    supports.
    """

    name: str
    display_name: str
    required_packages: tuple[str, ...] = ("torch", "transformers")

    def prepare(self, config: ExperimentConfig) -> None:
        """Fail early when optional ML packages are unavailable.

        The runner calls `prepare` before `run`, which means a missing PyTorch or
        Transformers install becomes a structured failed result instead of a
        cryptic import traceback inside a Slurm output file.
        """
        missing = [package for package in self.required_packages if find_spec(package) is None]
        if missing:
            package_list = ", ".join(missing)
            raise RuntimeError(
                f"{self.display_name} requires optional dependencies: {package_list}. "
                "Install them with: python -m pip install -e '.[workloads]'"
            )

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Reject execution for adapters that have not implemented a mode yet."""
        raise NotImplementedError(
            f"{self.display_name} adapter is scaffolded but execution is not implemented yet for {config.mode}."
        )


class GPT2SmallWorkload(TransformerWorkload):
    """Minimal Hugging Face/PyTorch GPT-2 workload for smoke runs and baseline plumbing.

    GPT-2 Small represents a decoder-only causal language model. The adapter
    supports a short synthetic-token training loop and a deterministic text
    generation path, both sized for smoke testing and local development rather
    than full model-quality training.
    """

    def __init__(self) -> None:
        """Initialise the adapter with the registry name used by GPT-2 configs."""
        super().__init__(name="gpt2_small", display_name="GPT-2 Small")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Dispatch GPT-2 configs to the supported training or inference path."""
        if config.mode == "training":
            return self._run_training(config)
        if config.mode == "inference":
            return self._run_inference(config)
        raise NotImplementedError(f"GPT-2 Small does not support mode: {config.mode}")

    def _run_training(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Run a compact causal-language-model training loop.

        The loop creates synthetic token IDs directly on the selected device.
        That avoids dataset downloads and keeps the benchmark focused on model
        execution, optimizer steps, result writing, and device compatibility.
        """
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
        # Create token IDs in the model vocabulary so the forward/backward pass
        # exercises the real embedding, transformer block, loss, and optimizer path.
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(batch_size, sequence_length),
            device=device,
        )

        # The measured region includes forward, loss, backward, and optimizer
        # work. Model loading and synthetic input creation are intentionally
        # outside this timer so repeated runs compare execution rather than setup.
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
        """Run the shared causal-LM inference helper for GPT-2 configs."""
        return _run_causal_lm_inference(config)


class DistilBERTWorkload(TransformerWorkload):
    """Minimal Hugging Face/PyTorch DistilBERT workload for encoder smoke runs.

    DistilBERT represents an encoder-style transformer. This adapter supports a
    short sequence-classification training loop and a single-sentence inference
    path so the project can compare encoder and decoder workload behavior.
    """

    def __init__(self) -> None:
        """Initialise the adapter with the registry name used by DistilBERT configs."""
        super().__init__(name="distilbert", display_name="DistilBERT")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Dispatch DistilBERT configs to the supported training or inference path."""
        if config.mode == "training":
            return self._run_training(config)
        if config.mode == "inference":
            return self._run_inference(config)
        raise NotImplementedError(f"DistilBERT does not support mode: {config.mode}")

    def _run_training(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Run a compact sequence-classification training loop.

        Inputs, masks, and labels are synthetic but shaped like a real
        classification batch. `ignore_mismatched_sizes=True` allows tiny smoke
        models and real model identifiers to share the same adapter even when a
        classification head must be freshly initialised.
        """
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
        # Keep tensors on the selected device from creation time to avoid timing
        # host-to-device copies inside the measured training loop.
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(batch_size, sequence_length),
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(low=0, high=num_labels, size=(batch_size,), device=device)

        # Time only the training step loop; model download/loading remains a
        # setup cost and is not part of the throughput calculation.
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
        """Run one DistilBERT sequence-classification inference pass.

        The output records latency and throughput for a single sample. For smoke
        testing this is enough to confirm tokenizer loading, model loading,
        device placement, and logits extraction.
        """
        torch, auto_model, auto_tokenizer = _load_sequence_classification_stack()
        parameters = config.workload.parameters
        device = _device(torch, parameters)
        dtype = _torch_dtype(torch, config.runtime.precision, device)
        model_name = str(parameters.get("hf_model_name", config.workload.model))
        text = str(parameters.get("text", "Transformer inference benchmark sentence."))
        num_labels = int(parameters.get("num_labels", 2))

        # Tokenization happens before timing because this adapter is measuring
        # model inference rather than end-to-end request preprocessing.
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
    """Minimal TinyLlama-style causal LM inference workload.

    TinyLlama is kept inference-only in this project because it gives a useful
    modern causal-LM serving smoke path without expanding local/HPC scope into
    full LLM training support.
    """

    def __init__(self) -> None:
        """Initialise the adapter with the registry name used by TinyLlama configs."""
        super().__init__(name="tinyllama", display_name="TinyLlama")

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Run TinyLlama inference configs and reject unsupported training configs."""
        if config.mode == "inference":
            return _run_causal_lm_inference(config)
        raise NotImplementedError("TinyLlama training is not implemented yet; use inference configs for this adapter.")


def transformer_workloads() -> list[TransformerWorkload]:
    """Return all transformer adapters registered by the default workload registry."""
    return [
        GPT2SmallWorkload(),
        DistilBERTWorkload(),
        TinyLlamaWorkload(),
    ]


def _load_causal_lm_stack() -> tuple[Any, Any, Any]:
    """Import the PyTorch and Hugging Face classes needed for causal LM workloads.

    Imports are intentionally delayed until execution so synthetic tests and CLI
    operations can run without optional transformer dependencies installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _load_sequence_classification_stack() -> tuple[Any, Any, Any]:
    """Import the PyTorch and Hugging Face classes needed for encoder classifiers."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _run_causal_lm_inference(config: ExperimentConfig) -> WorkloadRunOutput:
    """Run deterministic text generation for GPT-2-style and TinyLlama-style configs.

    The helper loads the configured tokenizer and model, encodes a prompt, and
    measures only the `generate` call. It reports prompt size, generated-token
    count, token throughput, and simple latency fields using the same schema
    expected by aggregation and dashboard code.
    """
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

    # Use greedy decoding for repeatable smoke and dev results. Sampling would
    # introduce extra randomness that is not useful for infrastructure checks.
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
    """Choose the execution device requested by config parameters.

    `auto` prefers CUDA for HPC, then Apple MPS for local Mac development, then
    CPU as the universal fallback. Explicit device requests fail loudly when the
    requested backend is unavailable so accidental CPU fallback does not hide a
    broken GPU environment.
    """
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
    """Return whether the imported PyTorch build exposes an available MPS backend."""
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def _torch_dtype(torch: Any, precision: str, device: Any) -> Any:
    """Map experiment precision strings to PyTorch dtypes supported by the device.

    CUDA runs can use fp16 or bf16 for GPU benchmarking. Local CPU/MPS dev runs
    fall back to fp32 for those precision labels because support can be partial
    or inconsistent, and the dev profile is about workflow validation rather
    than measuring true mixed-precision H100 performance.
    """
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if precision == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    raise NotImplementedError(f"Transformer workload does not support precision yet: {precision}")
