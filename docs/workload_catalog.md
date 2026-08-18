# Workload Catalog

# Performance Engineering of Distributed Transformer Training and Inference on Multi-GPU H100 Systems

## Purpose

This document defines the benchmark workloads that will be used throughout the dissertation. The workload selection aims to satisfy several objectives:

- Cover both compute-bound and memory-bound workloads.
- Represent realistic modern AI training and inference tasks.
- Use publicly available and reproducible software.
- Support strong and weak scaling studies.
- Support distributed training experiments.
- Enable comparison of different optimisation strategies.
- Remain feasible within the constraints of a shared HPC environment.

The workloads are organised into Primary, Secondary, and Stretch categories.

---

# Current Adapter Status

The executable framework currently distinguishes between validated smoke paths and full dissertation-scale runs. Smoke paths are intentionally small and are used to confirm configuration loading, runner behavior, result writing, and dependency wiring before moving to Slurm jobs.

| Workload adapter | Current state | Local smoke config | Notes |
| --- | --- | --- | --- |
| `synthetic` | Ready | `configs/local/synthetic_smoke.json` | CPU-only path used by tests and local validation. |
| `gpt2_small` | Implemented; cluster verification pending | `configs/local/gpt2_small_smoke.json` | Causal-LM training, PyTorch inference, vLLM routing, DDP/FSDP/ZeRO, precision, and profiling paths are implemented. |
| `distilbert` | Implemented; cluster verification pending | `configs/local/distilbert_smoke.json` | Classification training, eager/concurrent/data-parallel inference, SST-2 quality checks, and distributed training paths are implemented. |
| `tinyllama` | Implemented; optional full-size scope | `configs/local/tinyllama_inference_smoke.json` | Causal training and PyTorch/vLLM inference are implemented; full-size training remains optional for cost reasons. |

Full H100/L40S experiment runs still require the appropriate GPU software stack and cluster pilots. Semantic preflight rejects unsupported FP8, quantized, TensorRT-engine, and incompatible model/framework rows before scheduling; runtime failures are preserved as structured trial records.

---

# Workload Selection Criteria

Each workload was evaluated according to:

| Criterion | Importance |
|------------|------------|
| Open source | High |
| Reproducibility | High |
| Transformer-based | High |
| Multi-GPU support | High |
| Active community support | Medium |
| Industry relevance | High |
| Computational diversity | High |

---

# Primary Workloads

These workloads form the core of the dissertation and should be completed regardless of time constraints.

---

## W1 – NanoGPT

### Overview

NanoGPT is a lightweight implementation of GPT-style transformer training developed by Andrej Karpathy.

### Source

https://github.com/karpathy/nanoGPT

### Category

Transformer Training

### Workload Type

Compute-bound

### Purpose

- Baseline single-GPU training
- Strong scaling experiments
- Weak scaling experiments
- Precision studies

### Advantages

- Simple codebase
- Easy reproducibility
- Fast experimentation
- Well understood architecture

### Priority

Critical

---

## W2 – Hugging Face GPT-2

### Overview

GPT-2 provides a realistic transformer training workload using the Hugging Face ecosystem.

### Source

https://huggingface.co

### Category

Transformer Training

### Workload Type

Mixed compute and memory

### Purpose

- Distributed training evaluation
- FSDP evaluation
- DeepSpeed evaluation
- Precision experiments

### Priority

Critical

---

## W3 – Llama 3.2 1B

### Overview

Modern open-weight large language model suitable for multi-GPU experimentation.

### Source

https://huggingface.co/meta-llama

### Category

LLM Training and Inference

### Workload Type

Memory intensive

### Purpose

- Distributed training
- Memory scaling studies
- Inference benchmarking

### Priority

Critical

---

# Secondary Workloads

These workloads provide additional diversity and publication opportunities.

---

## W4 – BERT Base

### Source

https://huggingface.co

### Category

Natural Language Processing

### Workload Type

Moderately compute intensive

### Purpose

- Compare encoder vs decoder architectures
- Precision studies
- Distributed training

### Priority

High

---

## W5 – RoBERTa Base

### Source

https://huggingface.co

### Category

Transformer Training

### Workload Type

Mixed

### Purpose

- Cross-model validation
- Distributed training evaluation

### Priority

Medium

---

## W6 – Vision Transformer (ViT)

### Source

https://huggingface.co/google

### Category

Computer Vision

### Workload Type

Compute intensive

### Purpose

- Non-language transformer benchmark
- Compare scaling behaviour

### Priority

Medium

---

# Inference Workloads

These workloads are intended primarily for the optional inference section.

---

## W7 – Llama 3.2 Inference

### Purpose

- Latency measurement
- Throughput measurement
- Batch-size scaling

### Metrics

- Tokens/sec
- P50 latency
- P95 latency
- Memory usage

### Priority

High

---

## W8 – Quantised Llama 3.2

### Precision Modes

- FP16
- INT8
- 4-bit

### Purpose

- Quantisation analysis
- Memory reduction study
- Cost-efficiency analysis

### Priority

Medium

---

# HPC and Industry Benchmarks

These workloads help connect the project to existing benchmarking standards.

---

## W9 – DeepSpeed Benchmark Suite

### Source

https://github.com/microsoft/DeepSpeed

### Purpose

- ZeRO optimisation evaluation
- Memory scaling studies

### Priority

High

---

## W10 – MLPerf Training Benchmarks

### Source

https://mlcommons.org

### Purpose

- Industry-standard benchmarking
- External comparison point

### Priority

Medium

---

# Compute-Bound Workloads

The following workloads are expected to be primarily limited by GPU computation:

| Workload | Category |
|-----------|-----------|
| NanoGPT | Transformer Training |
| GPT-2 | Transformer Training |
| Vision Transformer | Computer Vision |
| BERT Base | NLP |

Research Goal:

- Measure GPU scaling efficiency.
- Measure arithmetic throughput.
- Measure impact of precision modes.

---

# Memory-Bound Workloads

The following workloads are expected to place significant pressure on GPU memory:

| Workload | Category |
|-----------|-----------|
| Llama 3.2 1B | LLM Training |
| Quantised Llama | LLM Inference |
| DeepSpeed ZeRO Experiments | Distributed Training |

Research Goal:

- Measure memory bottlenecks.
- Evaluate sharding strategies.
- Evaluate activation checkpointing.

---

# Dataset Catalog

---

## Dataset D1 – Tiny Shakespeare

### Source

https://github.com/karpathy/char-rnn

### Purpose

Rapid experimentation and debugging.

### Priority

High

---

## Dataset D2 – WikiText-2

### Source

https://huggingface.co/datasets

### Purpose

Language modelling benchmark.

### Priority

High

---

## Dataset D3 – WikiText-103

### Source

https://huggingface.co/datasets

### Purpose

Scaling studies.

### Priority

Medium

---

## Dataset D4 – OpenWebText

### Source

https://skylion007.github.io/OpenWebTextCorpus/

### Purpose

Realistic transformer training.

### Priority

Medium

---

## Dataset D5 – IMDB Reviews

### Source

https://huggingface.co/datasets

### Purpose

Classification experiments.

### Priority

Low

---

# Estimated Resource Requirements

| Workload | GPU Count | Estimated Duration |
|-----------|-----------|--------------------|
| NanoGPT | 1–4 | Low |
| GPT-2 | 1–4 | Medium |
| BERT Base | 1–4 | Medium |
| Llama 3.2 1B | 1–4 | High |
| ViT | 1–4 | Medium |

These estimates will be refined after baseline profiling.

---

# Recommended Execution Order

## Phase 1

1. NanoGPT
2. Tiny Shakespeare
3. Strong scaling
4. Weak scaling

## Phase 2

5. GPT-2
6. FSDP
7. DeepSpeed
8. Precision experiments

## Phase 3

9. Llama 3.2
10. Memory optimisation
11. Communication analysis

## Phase 4 (Stretch)

12. Inference benchmarking
13. Quantisation experiments
14. H100 vs L40S comparison

---

# Final Workload Set

## Mandatory

- NanoGPT
- GPT-2
- Llama 3.2 1B
- Tiny Shakespeare
- WikiText-2

## Recommended

- BERT Base
- DeepSpeed Benchmarks
- Vision Transformer

## Stretch

- MLPerf Benchmarks
- Quantised Inference
- H100 vs L40S Studies

---

# Expected Outcomes

The selected workloads should enable:

1. Strong and weak scaling analysis.
2. DDP, FSDP and DeepSpeed comparison.
3. Precision analysis across FP32, FP16, BF16 and FP8.
4. Communication overhead studies.
5. Data movement analysis.
6. Memory optimisation studies.
7. Inference performance benchmarking.
8. Optional H100 vs L40S comparison.
9. Publication-quality experimental results.
