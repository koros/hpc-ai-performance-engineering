# Project Plan

# Performance Engineering of Distributed Transformer Training and Inference on Multi-GPU H100 Systems

## MSc Artificial Intelligence Dissertation
### University of Plymouth

---

# Executive Summary

This project investigates the performance characteristics of modern transformer-based artificial intelligence workloads running on the University of Plymouth Lovelace High Performance Computing (HPC) facility.

The primary focus is distributed transformer training on NVIDIA H100 GPUs, with additional investigation of memory optimisation, communication overhead, scaling behaviour, precision formats, and inference performance.

The project combines concepts from:

- High Performance Computing (HPC)
- Artificial Intelligence
- Distributed Systems
- Performance Engineering
- GPU Computing

The final output will include a dissertation, reproducible benchmark suite, experiment datasets, performance dashboard, and potentially publication-quality research results.

---

# Project Vision

Modern AI models require increasingly large computational resources. While modern GPU systems provide significant computational capability, achieving efficient utilisation remains challenging due to communication overheads, memory constraints, and scaling inefficiencies.

The vision of this project is to produce practical and reproducible guidance for efficient transformer training and inference on modern multi-GPU systems.

---

# Research Aim

To evaluate and optimise the performance of distributed transformer training and inference workloads on modern GPU-accelerated HPC systems using the University of Plymouth Lovelace facility.

---

# Research Objectives

## Objective 1

Establish baseline performance measurements for transformer training and inference workloads.

### Deliverables

- Baseline benchmark suite
- Baseline performance report

---

## Objective 2

Measure strong and weak scaling behaviour across multiple H100 GPUs.

### Deliverables

- Scaling curves
- Parallel efficiency analysis

---

## Objective 3

Compare distributed training strategies.

### Technologies

- Distributed Data Parallel (DDP)
- Fully Sharded Data Parallel (FSDP)
- DeepSpeed ZeRO

### Deliverables

- Comparative analysis report
- Resource utilisation study

---

## Objective 4

Evaluate precision formats.

### Precision Modes

- FP32
- FP16
- BF16
- FP8

### Deliverables

- Precision-performance trade-off analysis

---

## Objective 5

Investigate communication and data movement bottlenecks.

### Deliverables

- Communication profile
- Data movement analysis

---

## Objective 6

Develop a performance visualisation dashboard.

### Deliverables

- Dashboard prototype
- Automated reporting pipeline

---

## Objective 7 (Stretch Goal)

Evaluate transformer inference workloads and compare H100 and L40S platforms.

---

# Research Questions

## RQ1

How efficiently do transformer workloads scale from one to four H100 GPUs?

## RQ2

Which distributed training strategy provides the best balance between throughput and memory efficiency?

## RQ3

What impact do different numerical precision formats have on performance and model quality?

## RQ4

What are the major communication bottlenecks limiting scaling efficiency?

## RQ5

How significant are data movement costs in distributed transformer training?

## RQ6

Can performance bottlenecks be automatically identified and visualised?

## RQ7

How do H100 and L40S platforms compare for AI workloads?

---

# Project Scope

## Included

- Transformer training
- Distributed training
- Multi-GPU scaling
- DDP
- FSDP
- DeepSpeed
- Precision optimisation
- Memory optimisation
- Communication profiling
- Dashboard development

## Optional

- Inference benchmarking
- Quantisation studies
- H100 vs L40S comparison

## Excluded

- Development of new transformer architectures
- Development of new distributed frameworks
- Large-scale multi-node cluster experiments

---

# Success Criteria

## Minimum Success

- Strong scaling completed
- Weak scaling completed
- DDP baseline completed
- FSDP comparison completed
- Dashboard prototype completed

---

## Target Success

- DeepSpeed comparison completed
- FP8 experiments completed
- Communication analysis completed
- Data movement analysis completed

---

## Stretch Success

- Inference benchmarking completed
- H100 versus L40S comparison completed
- Conference paper draft completed

---

# Proposal-Aligned Work Packages

| Work package | Proposal section | Matrix coverage | Main output |
|--------------|------------------|-----------------|-------------|
| WP1 | Environment validation | E001-E007 | Verified CUDA, PyTorch, NCCL, distributed launch, and inference environment |
| WP2 | Benchmark selection | Benchmark catalogue | Final workload set: GPT-2 Small, DistilBERT, TinyLlama |
| WP3 | Baseline profiling | E008-E019 | Single-GPU training and inference baselines with repeat variance |
| WP4 | Precision studies | E083-E096; E149-E200 selected inference precision runs | Precision-performance trade-off analysis |
| WP5 | Scaling studies | E032-E046; E035-E043 | Strong and weak scaling curves on 1, 2, and 4 H100 GPUs |
| WP6 | Distributed training strategies | E047-E062 | DDP, FSDP, and ZeRO comparison |
| WP7 | Memory optimisation | E063-E082 | Activation checkpointing, micro-batch, and accumulation analysis |
| WP8 | Data movement and communication | E097-E120 | DataLoader, pinned memory, prefetch, NCCL, and synchronisation bottleneck analysis |
| WP9 | Hardware comparison | E121-E128 | Selected H100 vs L40S comparison |
| WP10 | Inference studies | E129-E200 | Serving framework, batching, concurrency, prompt length, multi-GPU, and quantisation results |
| WP11 | Integrated analysis and dashboard | A01-A07 | Dashboard, bottleneck classification, GPU-hour and throughput/Watt synthesis |
| WP12 | Recommendations | A08 | Practical guidance for efficient training and serving |

---

# Timeline

## June

- Proposal submission
- Repository setup
- Literature review

## July

- Environment setup
- Baseline experiments
- Initial scaling experiments

## August

- Distributed training experiments
- Precision experiments
- Profiling experiments

## September

- Communication analysis
- Dashboard development
- Results analysis

## October

- Dissertation writing
- Final validation
- Submission preparation

---

# Risk Register

| Risk | Impact | Mitigation |
|--------|---------|------------|
| Delayed HPC access | High | Prepare local environment |
| GPU availability constraints | High | Prioritise critical experiments |
| Long queue times | Medium | Batch submissions |
| Failed jobs | Medium | Use checkpointing |
| Dataset issues | Low | Maintain alternative datasets |
| Software incompatibilities | Medium | Use version locking |
| Time limitations | High | Prioritise core experiments |

---

# Resource Requirements

## Hardware

Primary:

- 4 × NVIDIA H100 GPUs

Secondary:

- 8 × NVIDIA L40S GPUs (optional)

---

## Software

- PyTorch
- DeepSpeed
- Hugging Face
- CUDA
- NCCL
- Slurm
- JupyterLab

---

# Expected Outputs

## Research Outputs

- Dissertation
- Experiment matrix
- Benchmark datasets
- Scaling studies
- Performance analysis

---

## Software Outputs

- GitHub repository
- Experiment framework
- Dashboard
- Analysis scripts

---

## Publication Outputs

Primary paper:

Understanding Scaling Limits and Performance Bottlenecks of Distributed Transformer Training on Multi-GPU H100 Systems

Stretch paper:

Automated Performance Recommendations for Transformer Workloads on Modern GPU Systems

---

# Dissertation Chapter Mapping

| Dissertation Chapter | Related Work Package |
|----------------------|----------------------|
| Introduction | WP1 |
| Literature Review | WP1 |
| Methodology | WP2–WP7 |
| Results | WP3–WP7 |
| Discussion | WP3–WP8 |
| Conclusion | All Work Packages |

---

# Definition of Project Success

The project will be considered successful if it produces reproducible evidence explaining how transformer workloads scale on multi-GPU H100 systems and identifies practical optimisation strategies that improve performance, efficiency, and resource utilisation on modern AI infrastructure.
