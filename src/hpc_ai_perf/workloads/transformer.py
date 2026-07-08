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
        device = _device(torch)
        dtype = _torch_dtype(torch, config.runtime.precision, device)
        parameters = config.workload.parameters
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
        torch, auto_model, auto_tokenizer = _load_causal_lm_stack()
        device = _device(torch)
        dtype = _torch_dtype(torch, config.runtime.precision, device)
        parameters = config.workload.parameters
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


def transformer_workloads() -> list[TransformerWorkload]:
    return [
        GPT2SmallWorkload(),
        TransformerWorkload(name="distilbert", display_name="DistilBERT"),
        TransformerWorkload(name="tinyllama", display_name="TinyLlama"),
    ]


def _load_causal_lm_stack() -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, AutoModelForCausalLM, AutoTokenizer


def _device(torch: Any) -> Any:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_dtype(torch: Any, precision: str, device: Any) -> Any:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if precision == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    raise NotImplementedError(f"GPT-2 Small does not support precision yet: {precision}")
