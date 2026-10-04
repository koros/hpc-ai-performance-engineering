"""Untimed input preparation and placement evidence for scaling studies."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from array import array
from pathlib import Path
from typing import Any


def scaling_runtime_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in ("workloads/scaling.py", "workloads/transformer.py", "config/scaling.py", "profiling/measurement.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


class GlobalTrainingInputs:
    """The same ordered global examples, partitioned without duplication.

    Allocate and copy the entire small synthetic input stream before warm-up.
    Neither RNG, host-to-device copies nor hashing enter the measured loop.
    A fresh batch is consumed on every optimizer update, including warm-up.
    Classification workloads also draw one seeded label per sequence, after the
    tokens, so causal-LM inputs keep their original stream and digest.
    """

    def __init__(self, torch, config, runtime, device, vocab_size: int, task: str = "causal_lm", num_labels: int = 2):
        p = config.workload.parameters
        self.task = task
        self.micro_batch = p["per_gpu_batch_size"]
        self.accumulation = p["gradient_accumulation_steps"]
        global_batch = p["global_batch_size"]
        if global_batch != self.micro_batch * self.accumulation * runtime.world_size:
            raise ValueError("Observed world size violates the controlled global-batch contract")
        updates = p["warmup_steps"] + config.runtime.max_steps
        generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
        inputs = torch.randint(0, vocab_size, (updates, global_batch, p["sequence_length"]), generator=generator)
        digest = hashlib.sha256(inputs.numpy().tobytes())
        rank_batch = self.micro_batch * self.accumulation
        start = runtime.rank * rank_batch
        self.labels = None
        if task != "causal_lm":
            labels = torch.randint(0, num_labels, (updates, global_batch), generator=generator)
            digest.update(labels.numpy().tobytes())
            self.labels = labels[:, start : start + rank_batch].contiguous().to(device)
        self.sha256 = digest.hexdigest()
        self.inputs = inputs[:, start : start + rank_batch].contiguous().to(device)
        self.index = 0
        self.batches = updates * self.accumulation

    def next_batch(self):
        if self.index >= self.batches:
            raise RuntimeError("Controlled global input stream exhausted")
        update, micro = divmod(self.index, self.accumulation)
        self.index += 1
        start = micro * self.micro_batch
        tokens = self.inputs[update, start : start + self.micro_batch]
        if self.labels is None:
            return {"input_ids": tokens, "labels": tokens}
        return {
            "input_ids": tokens,
            "attention_mask": tokens.new_ones(tokens.shape),
            "labels": self.labels[update, start : start + self.micro_batch],
        }


MIN_DOCUMENT_TOKENS = 16
PAD_MULTIPLE = 8


def wikitext_documents(lines) -> list[str]:
    """Paragraphs only: WikiText headings (`= Title =`) and blank lines are not documents."""
    return [text for text in (line.strip() for line in lines) if text and not text.startswith("=")]


class RealTextInputs:
    """Variable-length real text, identical global content across GPU counts.

    Documents are taken in one seeded permutation of the whole corpus, so a
    larger global batch consumes a longer prefix of the same ordered stream.
    Every rank builds the full global structure, which lets it report padding
    and per-rank imbalance for all ranks without communication. Batching modes:

    - ``pad_to_max``: every document truncated and padded to sequence_length.
    - ``dynamic_pad``: each micro-batch padded only to its own longest document.
    - ``length_bucketed``: as dynamic_pad, but each global batch is sorted by
      length and its micro-batches dealt across ranks to balance work. The
      global batch content, and so the gradient, is unchanged.
    - ``pack``: documents joined with EOS and cut into full windows; no padding.
    """

    def __init__(self, torch, config, runtime, device, *, documents, tokenize, eos_id: int, measured_updates: int):
        p = config.workload.parameters
        self.mode = p["text_batching"]
        self.micro_batch = p["per_gpu_batch_size"]
        self.accumulation = p["gradient_accumulation_steps"]
        self.sequence_length = p["sequence_length"]
        global_batch = p["global_batch_size"]
        world = runtime.world_size
        if global_batch != self.micro_batch * self.accumulation * world:
            raise ValueError("Observed world size violates the controlled global-batch contract")
        updates = p["warmup_steps"] + config.runtime.max_steps
        order = torch.randperm(len(documents), generator=torch.Generator(device="cpu").manual_seed(config.runtime.seed))
        stream = (tokenize(documents[int(index)]) for index in order)
        if self.mode == "pack":
            global_samples = self._packed_windows(stream, updates * global_batch, eos_id)
        else:
            global_samples = self._documents(stream, updates * global_batch)
        digest = hashlib.sha256(f"{self.mode}:{self.sequence_length}".encode())
        for sample in global_samples:
            digest.update(len(sample).to_bytes(4, "little"))
            digest.update(array("i", sample).tobytes())
        self.sha256 = digest.hexdigest()

        # rank_batches[r][u] is rank r's ordered micro-batches (lists of samples) for update u.
        rank_batches = [[] for _ in range(world)]
        for update in range(updates):
            batch = global_samples[update * global_batch : (update + 1) * global_batch]
            for rank, micro_batches in enumerate(self._partition(batch, world)):
                rank_batches[rank].append(micro_batches)

        measured = range(updates - measured_updates, updates)
        self.useful_tokens = sum(len(s) for u in measured for r in range(world) for m in rank_batches[r][u] for s in m)
        rank_processed = [
            [sum(self._width(m) * len(m) for m in rank_batches[r][u]) for r in range(world)] for u in measured
        ]
        self.processed_tokens = sum(sum(row) for row in rank_processed)
        self.imbalance = sum(max(row) / (sum(row) / world) for row in rank_processed) / max(1, len(rank_processed))
        self.truncated_documents = self._truncated
        self.batches_list = [
            self._tensors(torch, micro, eos_id, device) for update in rank_batches[runtime.rank] for micro in update
        ]
        self.index = 0
        self.batches = len(self.batches_list)

    def _documents(self, stream, count: int) -> list[list[int]]:
        samples, self._truncated = [], 0
        for tokens in stream:
            if len(tokens) < MIN_DOCUMENT_TOKENS:
                continue
            if len(tokens) > self.sequence_length:
                self._truncated += 1
                tokens = tokens[: self.sequence_length]
            samples.append(list(tokens))
            if len(samples) == count:
                return samples
        raise RuntimeError(f"Real-text corpus exhausted after {len(samples)} of {count} documents")

    def _packed_windows(self, stream, count: int, eos_id: int) -> list[list[int]]:
        samples, buffer, self._truncated = [], [], 0
        for tokens in stream:
            if len(tokens) < MIN_DOCUMENT_TOKENS:
                continue
            buffer.extend(tokens)
            buffer.append(eos_id)
            while len(buffer) >= self.sequence_length:
                samples.append(buffer[: self.sequence_length])
                buffer = buffer[self.sequence_length :]
                if len(samples) == count:
                    return samples
        raise RuntimeError(f"Real-text corpus exhausted after {len(samples)} of {count} packed windows")

    def _partition(self, batch: list[list[int]], world: int) -> list[list[list[list[int]]]]:
        micro_count = world * self.accumulation
        if self.mode == "length_bucketed":
            ordered = sorted(batch, key=len)
            micros = [ordered[i * self.micro_batch : (i + 1) * self.micro_batch] for i in range(micro_count)]
            ranks: list[list[list[list[int]]]] = [[] for _ in range(world)]
            # Snake order hands each rank one long and one short micro-batch in turn.
            for index, micro in enumerate(micros):
                lap, position = divmod(index, world)
                ranks[position if lap % 2 == 0 else world - 1 - position].append(micro)
            return ranks
        rank_size = self.micro_batch * self.accumulation
        return [
            [
                batch[r * rank_size + m * self.micro_batch : r * rank_size + (m + 1) * self.micro_batch]
                for m in range(self.accumulation)
            ]
            for r in range(world)
        ]

    def _width(self, micro: list[list[int]]) -> int:
        if self.mode in {"pad_to_max", "pack"}:
            return self.sequence_length
        longest = max(len(sample) for sample in micro)
        return min(self.sequence_length, -(-longest // PAD_MULTIPLE) * PAD_MULTIPLE)

    def _tensors(self, torch, micro: list[list[int]], pad_id: int, device):
        width = self._width(micro)
        ids = torch.full((len(micro), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(micro), width), dtype=torch.long)
        for row, sample in enumerate(micro):
            ids[row, : len(sample)] = torch.tensor(sample, dtype=torch.long)
            mask[row, : len(sample)] = 1
        labels = ids.masked_fill(mask == 0, -100)
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}

    def next_batch(self):
        if self.index >= self.batches:
            raise RuntimeError("Controlled global input stream exhausted")
        batch = self.batches_list[self.index]
        self.index += 1
        return batch


def load_real_text_inputs(torch, config, runtime, device, model_name: str, revision: str | None, measured_updates: int):
    """Load the pinned WikiText-103 training split and the model's own tokenizer."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from hpc_ai_perf.config.scaling import REAL_TEXT_DATASET, REAL_TEXT_REVISION, REAL_TEXT_SUBSET

    data = load_dataset(REAL_TEXT_DATASET, REAL_TEXT_SUBSET, revision=REAL_TEXT_REVISION, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Real-text scaling requires a tokenizer with an EOS token")
    return RealTextInputs(
        torch,
        config,
        runtime,
        device,
        documents=wikitext_documents(data["text"]),
        tokenize=lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
        eos_id=eos_id,
        measured_updates=measured_updates,
    )


def capture_training_placement(torch: Any, runtime: Any) -> dict[str, Any]:
    from hpc_ai_perf.workloads.environment import _capture_gpu_topology

    props = torch.cuda.get_device_properties(runtime.local_rank)
    placement = {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "hostname": socket.gethostname(),
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "torch_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    ranks = [placement]
    if runtime.distributed:
        ranks = [None] * runtime.world_size
        torch.distributed.all_gather_object(ranks, placement)
    if runtime.rank != 0:
        return {}
    metrics = {**_capture_gpu_topology(), "training_rank_placement": json.dumps(ranks, sort_keys=True)}
    # Scheduler context is diagnostic and may be unavailable on non-Slurm hosts.
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--nodes", socket.gethostname(), "--format=%i|%t|%b|%C"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        metrics["scheduler_node_jobs_at_start"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        metrics["scheduler_node_jobs_at_start"] = f"unavailable: {type(exc).__name__}"
    return metrics
