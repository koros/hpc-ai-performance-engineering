# Experiment Matrix

# Performance Engineering of Distributed Transformer Training and Inference on Multi-GPU H100 Systems

## Overview

This document defines the planned experimental programme for the dissertation project. The experiments are organised into logical groups that progressively investigate scalability, efficiency, communication overheads, memory optimisation, precision formats, and inference performance on modern GPU systems.

The primary platform is the University of Plymouth Lovelace HPC facility, with a focus on NVIDIA H100 GPUs. Where resources and time permit, selected experiments will be extended to NVIDIA L40S GPUs.

---

# Success Criteria

## Minimum Deliverable

- Single GPU baseline established
- Strong scaling analysis completed
- Weak scaling analysis completed
- DDP vs FSDP comparison completed
- Performance dashboard prototype completed

## Target Deliverable

- DeepSpeed comparison completed
- FP8 experiments completed
- Communication profiling completed
- Data movement analysis completed

## Stretch Deliverable

- Inference benchmarking suite completed
- H100 vs L40S comparison completed
- Publication-ready paper draft completed

---

# A. Baseline Experiments

| ID | Experiment | Purpose | Key Metrics |
|----|------------|----------|-------------|
| A1 | Single GPU Training Baseline | Establish reference performance | Throughput, GPU Utilisation, Memory |
| A2 | Single GPU Inference Baseline | Establish inference reference | Tokens/sec, Latency |
| A3 | Resource Utilisation Profile | Identify bottlenecks | CPU, GPU, Memory |
| A4 | Model Size Sensitivity | Compare workload sizes | Throughput, Memory, Runtime |

---

# B. Scaling Experiments

## Strong Scaling

Fixed workload, increasing GPU count.

| ID | GPU Count |
|----|-----------|
| B1 | 1 GPU |
| B2 | 2 GPUs |
| B3 | 4 GPUs |

Metrics:

- Speedup
- Parallel efficiency
- Throughput
- GPU utilisation

## Weak Scaling

Workload increases proportionally with GPU count.

| ID | GPU Count |
|----|-----------|
| B4 | 1 GPU |
| B5 | 2 GPUs |
| B6 | 4 GPUs |

Metrics:

- Throughput per GPU
- Scaling efficiency
- Communication overhead

---

# C. Distributed Training Experiments

| ID | Strategy |
|----|----------|
| C1 | Distributed Data Parallel (DDP) |
| C2 | Fully Sharded Data Parallel (FSDP) |
| C3 | DeepSpeed ZeRO-1 |
| C4 | DeepSpeed ZeRO-2 |
| C5 | DeepSpeed ZeRO-3 |
| C6 | Pipeline Parallelism |
| C7 | Tensor Parallelism |

Metrics:

- Throughput
- GPU memory
- Communication overhead
- Runtime

---

# D. Precision Experiments

| ID | Precision |
|----|-----------|
| D1 | FP32 |
| D2 | FP16 |
| D3 | BF16 |
| D4 | FP8 |

Metrics:

- Throughput
- Memory consumption
- Validation loss
- Training stability

---

# E. Batch Size and Optimisation Experiments

| ID | Experiment |
|----|------------|
| E1 | Micro-batch size sweep |
| E2 | Gradient accumulation sweep |
| E3 | Global batch size scaling |
| E4 | Gradient clipping |

Metrics:

- Throughput
- Convergence rate
- Memory usage

---

# F. Optimiser Experiments

| ID | Optimiser |
|----|-----------|
| F1 | AdamW |
| F2 | Lion |

Additional Variables:

- Learning-rate scheduler
- Warmup strategy

Metrics:

- Final loss
- Convergence speed
- Runtime

---

# G. Memory Optimisation Experiments

| ID | Experiment |
|----|------------|
| G1 | Activation checkpointing |
| G2 | Parameter sharding |
| G3 | Sequence length scaling |
| G4 | KV cache analysis |

Metrics:

- Peak memory
- Throughput
- Runtime

---

# H. Data Movement Experiments

| ID | Experiment |
|----|------------|
| H1 | DataLoader worker tuning |
| H2 | Prefetching |
| H3 | Dataset shard sizing |
| H4 | Storage performance |
| H5 | CPU-GPU transfer analysis |

Metrics:

- GPU idle time
- I/O throughput
- Data transfer overhead

---

# I. Communication Experiments

| ID | Experiment |
|----|------------|
| I1 | NCCL communication profiling |
| I2 | All-reduce overhead analysis |
| I3 | Communication vs computation study |
| I4 | Multi-GPU synchronisation analysis |

Metrics:

- Communication time
- Synchronisation time
- Effective bandwidth

---

# J. Inference Experiments (Optional)

| ID | Experiment |
|----|------------|
| J1 | Batch size scaling |
| J2 | Prompt length scaling |
| J3 | Quantised inference |
| J4 | Throughput saturation analysis |

Metrics:

- Tokens/sec
- P50 latency
- P95 latency
- Memory usage

---

# K. Hardware Comparison Experiments (Optional)

| ID | Experiment |
|----|------------|
| K1 | H100 vs L40S Training |
| K2 | H100 vs L40S Inference |
| K3 | Cost Efficiency Comparison |
| K4 | Memory Efficiency Comparison |

Metrics:

- Throughput
- GPU hours
- Cost per workload
- Memory utilisation

---

# Standard Metrics

## Training Metrics

- Samples/sec
- Tokens/sec
- Runtime
- GPU utilisation
- Memory utilisation
- Validation loss

## Scaling Metrics

- Speedup
- Parallel efficiency
- Strong scaling efficiency
- Weak scaling efficiency

## Communication Metrics

- NCCL time
- All-reduce overhead
- Synchronisation overhead

## Inference Metrics

- Tokens/sec
- Latency (P50)
- Latency (P95)
- Memory usage

## Cost Metrics

- GPU hours
- Cost per experiment
- Cost per million tokens

---

# Expected Research Outputs

1. Strong scaling curves for transformer training.
2. Weak scaling curves for transformer training.
3. DDP, FSDP and DeepSpeed comparison.
4. Precision trade-off analysis.
5. Memory optimisation recommendations.
6. Data movement and communication bottleneck analysis.
7. Optional H100 vs L40S comparison.
8. Interactive performance dashboard.
9. Reproducible benchmark suite.
10. Publication-quality figures and datasets.
