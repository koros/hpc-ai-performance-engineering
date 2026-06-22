# Performance Engineering of Distributed Transformer Training and Inference on Multi-GPU H100 Systems

## Overview

This repository contains the research artefacts, experimental framework, benchmarking workloads, performance analysis tools, and dashboard developed as part of an MSc Artificial Intelligence dissertation at the University of Plymouth.

The project investigates how modern transformer-based AI workloads behave on large-scale GPU systems, with a particular focus on distributed training and inference on NVIDIA H100 GPUs within the Lovelace High Performance Computing (HPC) facility.

The research combines principles from:

- High Performance Computing (HPC)
- Deep Learning
- Distributed Systems
- Performance Engineering
- Large Language Models (LLMs)
- GPU Computing

The goal is to identify practical strategies that improve throughput, scalability, resource utilisation, and overall efficiency when training and serving transformer models on modern multi-GPU infrastructure.

---

## Research Motivation

Large transformer models have become the foundation of modern AI systems. However, training and deploying these models requires significant computational resources and can be prohibitively expensive.

Although modern hardware platforms provide enormous computational capabilities, achieving efficient utilisation remains a challenge due to:

- Communication overheads between GPUs
- Memory limitations
- Data movement bottlenecks
- Input pipeline inefficiencies
- Poor scaling beyond a small number of devices
- Complex distributed training strategies

This project aims to provide a systematic study of these factors and produce reproducible recommendations for researchers and practitioners.

---

## Research Objectives

### Primary Objectives

- Quantify training scalability on 1–4 NVIDIA H100 GPUs
- Measure strong and weak scaling efficiency
- Evaluate distributed training frameworks
- Identify major performance bottlenecks
- Measure communication and data movement costs
- Develop practical optimisation guidelines

### Secondary Objectives

- Evaluate transformer inference performance
- Investigate quantisation approaches
- Compare H100 and L40S GPU platforms
- Develop automated performance reporting tools

### Deliverables

- Reproducible experiment suite
- Benchmark dataset catalogue
- Performance dashboard
- Research dissertation
- Publication-quality figures and results
- Experiment recommendations and best practices

---

## Experimental Platform

Experiments are designed around the University of Plymouth Lovelace HPC facility.

### Hardware

- NVIDIA H100 SXM GPUs
- NVIDIA L40S GPUs (comparison studies)
- AMD EPYC CPUs
- High-speed GPU interconnects
- Shared parallel storage

### Software Stack

- Ubuntu Linux
- Slurm Workload Manager
- CUDA
- NCCL
- PyTorch
- DeepSpeed
- Hugging Face Transformers
- TensorBoard
- Weights & Biases (optional)
- JupyterLab

---

## Research Questions

1. How efficiently do transformer workloads scale from one to four H100 GPUs?
2. Which distributed training strategy provides the best performance-to-memory trade-off?
3. What impact do precision formats have on performance and model quality?
4. How significant are communication and data movement bottlenecks?
5. Which configuration achieves the highest throughput per GPU hour?
6. How do H100 and L40S platforms compare for AI workloads?
7. Can performance bottlenecks be automatically identified and visualised?

---

## Experiment Categories

### Baseline Experiments

- Single-GPU training baseline
- Single-GPU inference baseline
- Resource utilisation profiling
- Model size sensitivity analysis

### Scaling Experiments

- Strong scaling
- Weak scaling
- Multi-GPU efficiency analysis
- Communication overhead measurement

### Distributed Training Experiments

- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed ZeRO-1
- DeepSpeed ZeRO-2
- DeepSpeed ZeRO-3
- Pipeline Parallelism
- Tensor Parallelism

### Precision Experiments

- FP32
- FP16
- BF16
- FP8

### Memory Optimisation Experiments

- Activation checkpointing
- Gradient accumulation
- Sequence length scaling
- Model sharding

### Data Movement Experiments

- DataLoader optimisation
- Prefetching
- Dataset sharding
- Storage performance analysis

### Inference Experiments

- Batch size scaling
- Prompt length scaling
- Quantised inference
- Throughput vs latency analysis

### Hardware Comparison Experiments

- H100 vs L40S training
- H100 vs L40S inference
- Cost efficiency comparison
- Memory efficiency comparison

---

## Repository Structure

```text
hpc-ai-performance-engineering/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── proposal/
│   ├── dissertation/
│   ├── experiment_matrix/
│   └── papers/
│
├── configs/
│   ├── training/
│   ├── inference/
│   └── distributed/
│
├── experiments/
│   ├── baseline/
│   ├── scaling/
│   ├── precision/
│   ├── memory/
│   ├── communication/
│   └── inference/
│
├── src/
│   ├── training/
│   ├── inference/
│   ├── profiling/
│   └── analysis/
│
├── dashboard/
│
├── notebooks/
│
├── results/
│
└── slurm/
    ├── training/
    ├── profiling/
    └── benchmarking/
```

---

## Metrics Collected

### Training Metrics

- Samples per second
- Tokens per second
- Time per epoch
- GPU utilisation
- GPU memory utilisation
- Validation loss
- Training loss
- Convergence time

### Scaling Metrics

- Speedup
- Parallel efficiency
- Strong scaling efficiency
- Weak scaling efficiency

### Communication Metrics

- NCCL communication time
- All-reduce overhead
- Synchronisation delays
- Communication/computation ratio

### Inference Metrics

- Throughput
- P50 latency
- P95 latency
- Memory usage
- Tokens per second

### Cost Metrics

- GPU hours consumed
- Cost per experiment
- Cost per million tokens
- Throughput per GPU hour

---

## Dashboard

A custom dashboard will be developed to:

- Visualise scaling curves
- Compare experiment configurations
- Track resource utilisation
- Highlight bottlenecks
- Compare hardware platforms
- Generate publication-ready figures

---

## Reproducibility

The project emphasises reproducible research.

All experiments will include:

- Version-controlled source code
- Configuration files
- Experiment metadata
- Software environment specifications
- Hardware descriptions
- Logged performance metrics

---

## Potential Publications

### Primary Paper

Understanding Scaling Limits and Performance Bottlenecks of Distributed Transformer Training on Multi-GPU H100 Systems

### Secondary Paper (Stretch Goal)

Automated Performance Analysis and Optimisation Recommendations for Transformer Workloads on Modern GPU Systems

---

## Sustainable Development Goals

This work contributes indirectly to:

- SDG 4 – Quality Education
- SDG 9 – Industry, Innovation and Infrastructure
- SDG 12 – Responsible Consumption and Production
- SDG 13 – Climate Action

by promoting more efficient use of computational resources and reducing unnecessary energy consumption during AI development.

---

## Author

Geoffrey Koros

MSc Artificial Intelligence

University of Plymouth

---

## License

To be determined.
