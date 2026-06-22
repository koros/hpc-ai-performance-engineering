# Hardware Platform

# Performance Engineering of Distributed Transformer Training and Inference on Multi-GPU H100 Systems

## Purpose

This document describes the computing infrastructure used throughout the dissertation project. It serves as the authoritative reference for the hardware, software, networking, storage, and profiling environment used to conduct experiments.

This document also forms the basis of the Experimental Platform section of the final dissertation and any resulting publications.

---

# Lovelace High Performance Computing Facility

## Overview

The Lovelace High Performance Computing (HPC) facility at the University of Plymouth provides access to modern GPU-accelerated computing resources designed to support artificial intelligence, machine learning, scientific computing, and large-scale data analysis workloads.

The platform provides access to both NVIDIA H100 and NVIDIA L40S GPU systems, enabling evaluation of modern AI workloads across different accelerator architectures.

The facility is managed through a Slurm-based scheduling environment and provides access to shared storage and high-speed networking suitable for distributed computing.

---

# Research Goals Supported by the Platform

The Lovelace platform enables investigation of:

- Distributed transformer training
- Distributed transformer inference
- Strong scaling
- Weak scaling
- Communication bottlenecks
- Memory optimisation
- GPU utilisation analysis
- Precision optimisation
- Performance engineering of AI workloads

---

# Compute Infrastructure

## H100 Node

The primary experimental platform is the H100 node.

### GPU Configuration

| Component | Specification |
|------------|---------------|
| GPU Model | NVIDIA H100 SXM |
| GPU Count | 4 |
| Memory Per GPU | 80 GB |
| Total GPU Memory | 320 GB |
| Interconnect | NVLink |
| Intended Usage | AI Training and HPC Workloads |

### Why H100?

The H100 is currently one of the most advanced AI accelerators available and includes:

- Tensor Core acceleration
- FP16 support
- BF16 support
- FP8 support
- High memory bandwidth
- Multi-GPU NVLink connectivity

These features make it suitable for:

- Large Language Model training
- Distributed deep learning
- Scientific computing
- Mixed precision training

---

## L40S Node

The L40S platform will be used for optional comparison studies.

### GPU Configuration

| Component | Specification |
|------------|---------------|
| GPU Model | NVIDIA L40S |
| GPU Count | 8 |
| Memory Per GPU | 48 GB |
| Intended Usage | Inference and Mixed Workloads |

### Research Value

The L40S provides an opportunity to evaluate:

- Training efficiency
- Inference efficiency
- Cost effectiveness
- Memory constraints

relative to the H100 platform.

---

# CPU Infrastructure

## Processor Configuration

| Component | Specification |
|------------|---------------|
| CPU Family | AMD EPYC |
| CPU Count | 2 |
| Total CPU Cores | 64 |
| Purpose | Data loading, preprocessing, orchestration |

The CPUs play an important role in:

- Dataset loading
- Data preprocessing
- Distributed orchestration
- Input pipeline management

Several experiments will investigate whether workloads become CPU-bound or remain GPU-bound.

---

# Memory Infrastructure

## System Memory

| Component | Specification |
|------------|---------------|
| RAM Capacity | Approximately 1.5 TB |

Large system memory enables:

- Large dataset caching
- High-performance preprocessing
- Multiple concurrent workloads
- Efficient distributed training pipelines

---

# GPU Interconnect

## NVLink

The H100 GPUs are connected using NVLink.

### Purpose

NVLink provides high-bandwidth communication between GPUs and is essential for:

- Distributed Data Parallel training
- Fully Sharded Data Parallel training
- DeepSpeed communication
- Tensor parallelism
- Pipeline parallelism

### Research Relevance

Several experiments will quantify:

- Communication overhead
- Synchronisation cost
- Scaling efficiency
- Effective GPU utilisation

---

# Cluster Networking

## High-Speed Fabric

| Component | Specification |
|------------|---------------|
| Network Technology | NVIDIA NDR200 |
| Purpose | Distributed communication |

The high-speed network is designed to minimise communication overhead during:

- Multi-node workloads
- Distributed training
- Large data transfers

---

# Storage Infrastructure

## Shared Storage

| Component | Specification |
|------------|---------------|
| Storage Capacity | Approximately 2 PB |
| Access Type | Shared Cluster Storage |

### Research Relevance

Storage performance may influence:

- Dataset loading
- Training throughput
- GPU utilisation
- Data movement efficiency

Dedicated experiments will investigate storage bottlenecks and input pipeline performance.

---

# Software Environment

## Operating System

Linux-based HPC environment.

---

## Resource Manager

### Slurm

The cluster uses Slurm Workload Manager.

Responsibilities include:

- Job scheduling
- Resource allocation
- Queue management
- Multi-GPU job execution

Example workloads:

- Single GPU jobs
- Multi-GPU jobs
- Distributed training jobs
- Profiling jobs

---

# AI and Machine Learning Software Stack

## Deep Learning Frameworks

### PyTorch

Primary framework used throughout the project.

Applications:

- Model training
- Model evaluation
- Distributed training

---

### Hugging Face

Used for:

- Transformer models
- Datasets
- Training pipelines

---

### DeepSpeed

Used for:

- ZeRO optimisation
- Large model training
- Memory optimisation

---

# Communication Libraries

## CUDA

Provides GPU programming support.

---

## NCCL

Provides high-performance GPU communication.

Research experiments will profile:

- All-reduce operations
- Synchronisation overhead
- Communication bottlenecks

---

# Profiling and Monitoring Tools

## NVIDIA SMI

Used to collect:

- GPU utilisation
- Memory utilisation
- Power metrics (where available)

---

## PyTorch Profiler

Used to analyse:

- Kernel execution
- Data loading
- Communication overhead
- Training bottlenecks

---

## Nsight Systems

Used for:

- Timeline analysis
- GPU activity analysis
- Communication profiling

---

## Nsight Compute

Used for:

- Kernel analysis
- Hardware efficiency analysis

---

# Experimental Constraints

The platform is a shared academic resource.

Potential constraints include:

- Queue waiting times
- Resource allocation limits
- Concurrent users
- Time restrictions on jobs

The experimental design therefore prioritises:

- Reproducibility
- Efficient resource utilisation
- Incremental experimentation
- Careful scheduling

---

# Platform Risks

| Risk | Mitigation |
|--------|------------|
| Limited GPU availability | Prioritise critical experiments |
| Long queue times | Batch submissions |
| Storage limitations | Dataset management |
| Software version changes | Environment capture |
| Failed jobs | Automated checkpointing |

---

# Reproducibility Strategy

All experiments should capture:

- Hardware information
- Software versions
- CUDA version
- NCCL version
- PyTorch version
- Git commit hash
- Experiment configuration

This information will be stored alongside experiment results.

---

# Platform Summary

## Primary Platform

- 4 × NVIDIA H100 SXM GPUs
- 80 GB memory per GPU
- NVLink connectivity
- AMD EPYC processors
- 1.5 TB RAM

## Secondary Platform

- 8 × NVIDIA L40S GPUs

## Supporting Infrastructure

- NVIDIA NDR200 networking
- 2 PB shared storage
- Slurm scheduling
- CUDA
- NCCL
- PyTorch
- DeepSpeed

---

# Expected Research Outcomes Enabled by the Platform

The Lovelace facility enables:

1. Strong scaling studies.
2. Weak scaling studies.
3. Distributed training evaluation.
4. DeepSpeed optimisation studies.
5. FSDP evaluation.
6. Communication bottleneck analysis.
7. Data movement analysis.
8. Inference benchmarking.
9. H100 versus L40S comparison.
10. Publication-quality performance engineering research.
