# Comprehensive Experiment Matrix

Source: `Appendix_A_Comprehensive_Experiment_Matrix.docx (original planning document; the matrix below is the public specification)`.

This document is the planning source of truth for the dissertation experiment programme. Training remains the core scope; inference is retained as a high-impact secondary scope and protected by priority levels.

## Proposal Alignment

| Proposal section | Matrix coverage |
| --- | --- |
| Environment validation | E001-E007 |
| Benchmark selection | Benchmark catalogue and factorial design dimensions |
| Baseline profiling | E008-E019 |
| Precision studies | E083-E096 and E149-E200 selected inference precision runs |
| Scaling studies | E032-E046 strong scaling and E035-E043 weak scaling |
| Distributed training strategies | E047-E062 DDP, FSDP, ZeRO-1, ZeRO-2, ZeRO-3 |
| Memory optimisation | E063-E082 activation checkpointing, micro-batch and accumulation sweeps |
| Data movement and communication | E097-E120 DataLoader, pinned memory, prefetching, NCCL and FSDP communication profiling |
| Hardware comparison | E121-E128 H100 vs L40S |
| Inference studies | E006-E007, E010-E011, E014-E015, E018-E019, E023, E027, E031, E087-E096, E129-E200 |
| Integrated analysis and dashboard | A01-A07 |
| Recommendations | A08 deployment recommendations |

## Priority Strategy

| Priority | Meaning | Treatment |
| --- | --- | --- |
| Must Have | Required for a complete dissertation | Run first and repeat where needed. These experiments answer the main research questions. |
| Should Have | Important for stronger impact | Run after core results are stable. These add depth, especially for inference and hardware comparison. |
| Optional | Stretch goal | Run only if time, software stability, and HPC access allow. |

## Benchmark Catalogue

| Workload | Type | Primary use | Reason for inclusion |
| --- | --- | --- | --- |
| GPT-2 Small | Decoder Transformer | Training and inference | Fast, reproducible, widely used baseline for transformer language modelling. |
| DistilBERT | Encoder Transformer | Training and inference | Provides a different transformer family and may show different memory/compute behaviour. |
| TinyLlama | Small modern LLM | Primarily inference; optional training | More representative of modern LLM serving and prompt-based workloads. |

## Factorial Design Dimensions

| Factor group | Values |
| --- | --- |
| Workload | GPT-2 Small; DistilBERT; TinyLlama |
| GPU count | 1, 2, 4 GPUs on H100; selected 1-GPU L40S comparison |
| Training strategy | Single GPU; DDP; FSDP; ZeRO-1; ZeRO-2; ZeRO-3 |
| Precision | FP32; FP16; BF16; FP8; selected quantized inference |
| Memory optimisation | Activation checkpointing; micro-batch size; gradient accumulation |
| Data movement | DataLoader workers; pinned memory; prefetch factor; CPU-to-GPU transfer |
| Inference framework | PyTorch eager; vLLM; TensorRT-LLM |
| Serving load | Batch size; concurrent requests; prompt length; tensor parallel inference |

## Measured Variables

| Metric group | Measurements |
| --- | --- |
| Training performance | Runtime, samples/sec, tokens/sec, training loss, validation loss, GPU-hours |
| Resource utilisation | GPU utilisation, GPU memory, CPU utilisation, data-loading time |
| Scalability | Speedup, strong-scaling efficiency, weak-scaling efficiency, throughput retention |
| Communication | NCCL bandwidth, AllReduce time, synchronisation overhead, NVLink utilisation, GPU idle time |
| Inference | Tokens/sec, time-to-first-token, p50 latency, p95 latency, memory, concurrent request throughput |
| Energy and efficiency | Power draw, energy consumption, throughput/GPU-hour, throughput/Watt |

## Estimated GPU-Hour Budget

| Experiment group | Estimated GPU-hours |
| --- | --- |
| Environment validation | 4.3 |
| Baseline profiling | 8.6 |
| Workload characterisation | 7.2 |
| Strong scaling | 12.2 |
| Weak scaling | 7.6 |
| Distributed training strategy | 24.0 |
| Memory optimisation | 14.4 |
| Precision study | 9.4 |
| Data movement | 6.4 |
| Communication analysis | 9.6 |
| Hardware comparison | 5.2 |
| Inference framework | 6.0 |
| Inference batching | 7.8 |
| Inference concurrency | 6.5 |
| Inference prompt length | 5.2 |
| Inference multi-GPU | 7.2 |
| Inference precision/quantization | 6.0 |
| Total experiment estimate | 147.6 |

## Budget by Priority

| Priority | Estimated GPU-hours |
| --- | --- |
| Must Have | 89.4 |
| Should Have | 46.1 |
| Optional | 12.1 |

## Environment Validation

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E001 | Environment validation | All | Training | CUDA/PyTorch | 1 | N/A | Confirm CUDA, driver, PyTorch, and GPU visibility before experiments | GPU detected, CUDA version, smoke test runtime | Must Have | 0.5 |
| E002 | Environment validation | All | Training | PyTorch DDP/NCCL | 2 | N/A | Confirm two-GPU distributed launch and NCCL communication | AllReduce success, bandwidth, errors | Must Have | 0.5 |
| E003 | Environment validation | All | Training | PyTorch DDP/NCCL | 4 | N/A | Confirm four-GPU distributed launch and NCCL communication | AllReduce success, bandwidth, errors | Must Have | 1.0 |
| E004 | Environment validation | All | Training | FSDP | 2 | N/A | Confirm FSDP script executes correctly on two GPUs | Successful run, memory trace | Must Have | 0.5 |
| E005 | Environment validation | All | Training | DeepSpeed ZeRO | 2 | N/A | Confirm DeepSpeed launcher and ZeRO configuration work | Successful run, memory trace | Should Have | 0.5 |
| E006 | Environment validation | All | Inference | vLLM | 1 | N/A | Confirm vLLM serving and benchmark tooling work | Requests served, tokens/sec, TTFT | Must Have | 0.5 |
| E007 | Environment validation | All | Inference | TensorRT-LLM | 1 | N/A | Confirm TensorRT-LLM environment if available | Engine build, served requests | Optional | 0.8 |

## Baseline Profiling

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E008 | Baseline profiling | GPT-2 Small | Training | PyTorch single GPU | 1 | BF16 | Establish single-GPU training baseline for decoder | Throughput, runtime, memory, loss, GPU util, GPU-hours | Must Have | 1.0 |
| E009 | Baseline profiling | GPT-2 Small | Training | PyTorch single GPU repeat | 1 | BF16 | Repeat baseline to estimate run-to-run variance | Mean, std dev, coefficient of variation | Must Have | 0.7 |
| E010 | Baseline profiling | GPT-2 Small | Inference | PyTorch eager | 1 | BF16 | Establish single-GPU inference baseline | Tokens/sec, latency, TTFT, memory | Must Have | 0.5 |
| E011 | Baseline profiling | GPT-2 Small | Inference | PyTorch repeat | 1 | BF16 | Repeat inference baseline to estimate variance | Mean, p50, p95, std dev | Must Have | 0.4 |
| E012 | Baseline profiling | DistilBERT | Training | PyTorch single GPU | 1 | BF16 | Establish single-GPU training baseline for encoder | Throughput, runtime, memory, loss, GPU util, GPU-hours | Must Have | 1.0 |
| E013 | Baseline profiling | DistilBERT | Training | PyTorch single GPU repeat | 1 | BF16 | Repeat baseline to estimate run-to-run variance | Mean, std dev, coefficient of variation | Must Have | 0.7 |
| E014 | Baseline profiling | DistilBERT | Inference | PyTorch eager | 1 | BF16 | Establish single-GPU inference baseline | Tokens/sec, latency, TTFT, memory | Must Have | 0.5 |
| E015 | Baseline profiling | DistilBERT | Inference | PyTorch repeat | 1 | BF16 | Repeat inference baseline to estimate variance | Mean, p50, p95, std dev | Must Have | 0.4 |
| E016 | Baseline profiling | TinyLlama | Training | PyTorch single GPU | 1 | BF16 | Establish single-GPU training baseline for small LLM | Throughput, runtime, memory, loss, GPU util, GPU-hours | Should Have | 1.5 |
| E017 | Baseline profiling | TinyLlama | Training | PyTorch single GPU repeat | 1 | BF16 | Repeat baseline to estimate run-to-run variance | Mean, std dev, coefficient of variation | Should Have | 1.0 |
| E018 | Baseline profiling | TinyLlama | Inference | PyTorch eager | 1 | BF16 | Establish single-GPU inference baseline | Tokens/sec, latency, TTFT, memory | Must Have | 0.5 |
| E019 | Baseline profiling | TinyLlama | Inference | PyTorch repeat | 1 | BF16 | Repeat inference baseline to estimate variance | Mean, p50, p95, std dev | Must Have | 0.4 |

## Workload Characterisation

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E020 | Workload characterisation | GPT-2 Small | Training | Profiler | 1 | BF16 | Identify compute intensity and kernel utilisation | GPU util, SM occupancy proxy, kernel time | Must Have | 0.7 |
| E021 | Workload characterisation | GPT-2 Small | Training | Memory profiler | 1 | BF16 | Identify memory pressure and peak memory behaviour | Peak memory, allocation pattern, OOM risk | Must Have | 0.7 |
| E022 | Workload characterisation | GPT-2 Small | Training | Data pipeline profiler | 1 | BF16 | Identify data-loading or preprocessing stalls | CPU util, batch load time, GPU idle time | Must Have | 0.5 |
| E023 | Workload characterisation | GPT-2 Small | Inference | vLLM/PyTorch profiler | 1 | BF16 | Characterise inference as latency, memory, or compute constrained | TTFT, tokens/sec, memory, GPU util | Should Have | 0.5 |
| E024 | Workload characterisation | DistilBERT | Training | Profiler | 1 | BF16 | Identify compute intensity and kernel utilisation | GPU util, SM occupancy proxy, kernel time | Must Have | 0.7 |
| E025 | Workload characterisation | DistilBERT | Training | Memory profiler | 1 | BF16 | Identify memory pressure and peak memory behaviour | Peak memory, allocation pattern, OOM risk | Must Have | 0.7 |
| E026 | Workload characterisation | DistilBERT | Training | Data pipeline profiler | 1 | BF16 | Identify data-loading or preprocessing stalls | CPU util, batch load time, GPU idle time | Must Have | 0.5 |
| E027 | Workload characterisation | DistilBERT | Inference | vLLM/PyTorch profiler | 1 | BF16 | Characterise inference as latency, memory, or compute constrained | TTFT, tokens/sec, memory, GPU util | Should Have | 0.5 |
| E028 | Workload characterisation | TinyLlama | Training | Profiler | 1 | BF16 | Identify compute intensity and kernel utilisation | GPU util, SM occupancy proxy, kernel time | Must Have | 0.7 |
| E029 | Workload characterisation | TinyLlama | Training | Memory profiler | 1 | BF16 | Identify memory pressure and peak memory behaviour | Peak memory, allocation pattern, OOM risk | Must Have | 0.7 |
| E030 | Workload characterisation | TinyLlama | Training | Data pipeline profiler | 1 | BF16 | Identify data-loading or preprocessing stalls | CPU util, batch load time, GPU idle time | Must Have | 0.5 |
| E031 | Workload characterisation | TinyLlama | Inference | vLLM/PyTorch profiler | 1 | BF16 | Characterise inference as latency, memory, or compute constrained | TTFT, tokens/sec, memory, GPU util | Should Have | 0.5 |

## Strong Scaling

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E032 | Strong scaling | GPT-2 Small | Training | DDP | 1 | BF16 | Train fixed workload while increasing GPU count | Runtime, speedup, parallel efficiency, throughput | Must Have | 0.8 |
| E033 | Strong scaling | GPT-2 Small | Training | DDP | 2 | BF16 | Train fixed workload while increasing GPU count | Runtime, speedup, parallel efficiency, throughput | Must Have | 1.0 |
| E034 | Strong scaling | GPT-2 Small | Training | DDP | 4 | BF16 | Train fixed workload while increasing GPU count | Runtime, speedup, parallel efficiency, throughput | Must Have | 2.0 |
| E038 | Strong scaling | DistilBERT | Training | DDP | 1 | BF16 | Train fixed workload while increasing GPU count | Runtime, speedup, parallel efficiency, throughput | Must Have | 0.8 |
| E039 | Strong scaling | DistilBERT | Training | DDP | 2 | BF16 | Train fixed workload while increasing GPU count | Runtime, speedup, parallel efficiency, throughput | Must Have | 1.0 |
| E040 | Strong scaling | DistilBERT | Training | DDP | 4 | BF16 | Train fixed workload while increasing GPU count | Runtime, speedup, parallel efficiency, throughput | Must Have | 2.0 |
| E044 | Strong scaling | TinyLlama | Training | DDP or FSDP | 1 | BF16 | Optional fixed-workload training scaling if time permits | Runtime, speedup, memory, OOM risk | Optional | 1.0 |
| E045 | Strong scaling | TinyLlama | Training | DDP or FSDP | 2 | BF16 | Optional fixed-workload training scaling if time permits | Runtime, speedup, memory, OOM risk | Optional | 1.2 |
| E046 | Strong scaling | TinyLlama | Training | DDP or FSDP | 4 | BF16 | Optional fixed-workload training scaling if time permits | Runtime, speedup, memory, OOM risk | Optional | 2.4 |

## Weak Scaling

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E035 | Weak scaling | GPT-2 Small | Training | DDP | 1 | BF16 | Increase workload with GPU count and measure throughput retention | Throughput/GPU, efficiency, memory, loss | Must Have | 0.8 |
| E036 | Weak scaling | GPT-2 Small | Training | DDP | 2 | BF16 | Increase workload with GPU count and measure throughput retention | Throughput/GPU, efficiency, memory, loss | Must Have | 1.0 |
| E037 | Weak scaling | GPT-2 Small | Training | DDP | 4 | BF16 | Increase workload with GPU count and measure throughput retention | Throughput/GPU, efficiency, memory, loss | Must Have | 2.0 |
| E041 | Weak scaling | DistilBERT | Training | DDP | 1 | BF16 | Increase workload with GPU count and measure throughput retention | Throughput/GPU, efficiency, memory, loss | Must Have | 0.8 |
| E042 | Weak scaling | DistilBERT | Training | DDP | 2 | BF16 | Increase workload with GPU count and measure throughput retention | Throughput/GPU, efficiency, memory, loss | Must Have | 1.0 |
| E043 | Weak scaling | DistilBERT | Training | DDP | 4 | BF16 | Increase workload with GPU count and measure throughput retention | Throughput/GPU, efficiency, memory, loss | Must Have | 2.0 |

## Distributed Training Strategy

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E047 | Distributed training strategy | GPT-2 Small | Training | DDP | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 1.2 |
| E048 | Distributed training strategy | GPT-2 Small | Training | DDP | 4 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 2.4 |
| E049 | Distributed training strategy | GPT-2 Small | Training | FSDP full shard | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 1.2 |
| E050 | Distributed training strategy | GPT-2 Small | Training | FSDP full shard | 4 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 2.4 |
| E051 | Distributed training strategy | GPT-2 Small | Training | FSDP sharded grad | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Should Have | 1.2 |
| E052 | Distributed training strategy | GPT-2 Small | Training | ZeRO-1 | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Should Have | 1.2 |
| E053 | Distributed training strategy | GPT-2 Small | Training | ZeRO-2 | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Should Have | 1.2 |
| E054 | Distributed training strategy | GPT-2 Small | Training | ZeRO-3 | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Optional | 1.2 |
| E055 | Distributed training strategy | DistilBERT | Training | DDP | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 1.2 |
| E056 | Distributed training strategy | DistilBERT | Training | DDP | 4 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 2.4 |
| E057 | Distributed training strategy | DistilBERT | Training | FSDP full shard | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 1.2 |
| E058 | Distributed training strategy | DistilBERT | Training | FSDP full shard | 4 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Must Have | 2.4 |
| E059 | Distributed training strategy | DistilBERT | Training | FSDP sharded grad | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Should Have | 1.2 |
| E060 | Distributed training strategy | DistilBERT | Training | ZeRO-1 | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Should Have | 1.2 |
| E061 | Distributed training strategy | DistilBERT | Training | ZeRO-2 | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Should Have | 1.2 |
| E062 | Distributed training strategy | DistilBERT | Training | ZeRO-3 | 2 | BF16 | Compare distributed strategy against baseline | Throughput, memory, comm overhead, GPU-hours | Optional | 1.2 |

## Memory Optimisation

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E063 | Memory optimisation | GPT-2 Small | Training | FSDP | 2 | BF16 | Compare activation checkpointing OFF | Peak memory, throughput, runtime, loss | Must Have | 1.0 |
| E064 | Memory optimisation | GPT-2 Small | Training | FSDP | 2 | BF16 | Compare activation checkpointing ON | Peak memory, throughput, runtime, loss | Must Have | 1.0 |
| E065 | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Sweep micro-batch size 8 | Throughput, memory, OOM, loss stability | Should Have | 0.6 |
| E066 | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Sweep micro-batch size 16 | Throughput, memory, OOM, loss stability | Must Have | 0.6 |
| E067 | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Sweep micro-batch size 32 | Throughput, memory, OOM, loss stability | Must Have | 0.6 |
| E068 | Memory optimisation | GPT-2 Small | Training | DDP | 1 | BF16 | Sweep micro-batch size 64 | Throughput, memory, OOM, loss stability | Should Have | 0.6 |
| E069 | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Gradient accumulation steps 1 | Effective batch, throughput, memory, stability | Should Have | 0.7 |
| E070 | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Gradient accumulation steps 2 | Effective batch, throughput, memory, stability | Optional | 0.7 |
| E071 | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Gradient accumulation steps 4 | Effective batch, throughput, memory, stability | Should Have | 0.7 |
| E072 | Memory optimisation | GPT-2 Small | Training | DDP | 2 | BF16 | Gradient accumulation steps 8 | Effective batch, throughput, memory, stability | Optional | 0.7 |
| E073 | Memory optimisation | DistilBERT | Training | FSDP | 2 | BF16 | Compare activation checkpointing OFF | Peak memory, throughput, runtime, loss | Must Have | 1.0 |
| E074 | Memory optimisation | DistilBERT | Training | FSDP | 2 | BF16 | Compare activation checkpointing ON | Peak memory, throughput, runtime, loss | Must Have | 1.0 |
| E075 | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Sweep micro-batch size 8 | Throughput, memory, OOM, loss stability | Should Have | 0.6 |
| E076 | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Sweep micro-batch size 16 | Throughput, memory, OOM, loss stability | Must Have | 0.6 |
| E077 | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Sweep micro-batch size 32 | Throughput, memory, OOM, loss stability | Must Have | 0.6 |
| E078 | Memory optimisation | DistilBERT | Training | DDP | 1 | BF16 | Sweep micro-batch size 64 | Throughput, memory, OOM, loss stability | Should Have | 0.6 |
| E079 | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Gradient accumulation steps 1 | Effective batch, throughput, memory, stability | Should Have | 0.7 |
| E080 | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Gradient accumulation steps 2 | Effective batch, throughput, memory, stability | Optional | 0.7 |
| E081 | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Gradient accumulation steps 4 | Effective batch, throughput, memory, stability | Should Have | 0.7 |
| E082 | Memory optimisation | DistilBERT | Training | DDP | 2 | BF16 | Gradient accumulation steps 8 | Effective batch, throughput, memory, stability | Optional | 0.7 |

## Precision Study

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E083 | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | FP32 | Compare training precision FP32 | Throughput, memory, power, loss, stability | Must Have | 0.8 |
| E084 | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | FP16 | Compare training precision FP16 | Throughput, memory, power, loss, stability | Must Have | 0.8 |
| E085 | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | BF16 | Compare training precision BF16 | Throughput, memory, power, loss, stability | Must Have | 0.8 |
| E086 | Precision study | GPT-2 Small | Training | PyTorch/Transformer Engine | 1 | FP8 | Compare training precision FP8 | Throughput, memory, power, loss, stability | Should Have | 0.8 |
| E087 | Precision study | GPT-2 Small | Inference | vLLM/PyTorch | 1 | FP16 | Compare inference precision FP16 | Tokens/sec, TTFT, p95 latency, memory, power | Must Have | 0.5 |
| E088 | Precision study | GPT-2 Small | Inference | vLLM/PyTorch | 1 | BF16 | Compare inference precision BF16 | Tokens/sec, TTFT, p95 latency, memory, power | Must Have | 0.5 |
| E089 | Precision study | GPT-2 Small | Inference | vLLM/PyTorch | 1 | FP8 | Compare inference precision FP8 | Tokens/sec, TTFT, p95 latency, memory, power | Should Have | 0.5 |
| E090 | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | FP32 | Compare training precision FP32 | Throughput, memory, power, loss, stability | Must Have | 0.8 |
| E091 | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | FP16 | Compare training precision FP16 | Throughput, memory, power, loss, stability | Must Have | 0.8 |
| E092 | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | BF16 | Compare training precision BF16 | Throughput, memory, power, loss, stability | Must Have | 0.8 |
| E093 | Precision study | DistilBERT | Training | PyTorch/Transformer Engine | 1 | FP8 | Compare training precision FP8 | Throughput, memory, power, loss, stability | Should Have | 0.8 |
| E094 | Precision study | DistilBERT | Inference | vLLM/PyTorch | 1 | FP16 | Compare inference precision FP16 | Tokens/sec, TTFT, p95 latency, memory, power | Must Have | 0.5 |
| E095 | Precision study | DistilBERT | Inference | vLLM/PyTorch | 1 | BF16 | Compare inference precision BF16 | Tokens/sec, TTFT, p95 latency, memory, power | Must Have | 0.5 |
| E096 | Precision study | DistilBERT | Inference | vLLM/PyTorch | 1 | FP8 | Compare inference precision FP8 | Tokens/sec, TTFT, p95 latency, memory, power | Should Have | 0.5 |

## Data Movement

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E097 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | DataLoader num_workers=2 | Batch load time, GPU idle, CPU util, throughput | Should Have | 0.4 |
| E098 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | DataLoader num_workers=4 | Batch load time, GPU idle, CPU util, throughput | Must Have | 0.4 |
| E099 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | DataLoader num_workers=8 | Batch load time, GPU idle, CPU util, throughput | Must Have | 0.4 |
| E100 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | DataLoader num_workers=16 | Batch load time, GPU idle, CPU util, throughput | Should Have | 0.4 |
| E101 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Evaluate pinned_memory OFF | CPU-to-GPU transfer, GPU idle, throughput | Should Have | 0.4 |
| E102 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Evaluate pinned_memory ON | CPU-to-GPU transfer, GPU idle, throughput | Must Have | 0.4 |
| E103 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Evaluate prefetch_factor low | CPU-to-GPU transfer, GPU idle, throughput | Should Have | 0.4 |
| E104 | Data movement | GPT-2 Small | Training | DataLoader | 1 | BF16 | Evaluate prefetch_factor high | CPU-to-GPU transfer, GPU idle, throughput | Must Have | 0.4 |
| E105 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | DataLoader num_workers=2 | Batch load time, GPU idle, CPU util, throughput | Should Have | 0.4 |
| E106 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | DataLoader num_workers=4 | Batch load time, GPU idle, CPU util, throughput | Must Have | 0.4 |
| E107 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | DataLoader num_workers=8 | Batch load time, GPU idle, CPU util, throughput | Must Have | 0.4 |
| E108 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | DataLoader num_workers=16 | Batch load time, GPU idle, CPU util, throughput | Should Have | 0.4 |
| E109 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Evaluate pinned_memory OFF | CPU-to-GPU transfer, GPU idle, throughput | Should Have | 0.4 |
| E110 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Evaluate pinned_memory ON | CPU-to-GPU transfer, GPU idle, throughput | Must Have | 0.4 |
| E111 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Evaluate prefetch_factor low | CPU-to-GPU transfer, GPU idle, throughput | Should Have | 0.4 |
| E112 | Data movement | DistilBERT | Training | DataLoader | 1 | BF16 | Evaluate prefetch_factor high | CPU-to-GPU transfer, GPU idle, throughput | Must Have | 0.4 |

## Communication Analysis

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E113 | Communication analysis | GPT-2 Small | Training | NCCL/DDP | 2 | BF16 | Profile distributed gradient synchronization | AllReduce time, bandwidth, comm/compute ratio | Must Have | 0.8 |
| E114 | Communication analysis | GPT-2 Small | Training | FSDP/NCCL | 2 | BF16 | Profile sharded communication and memory exchange | Comm overhead, idle time, NVLink utilisation | Must Have | 0.8 |
| E115 | Communication analysis | GPT-2 Small | Training | NCCL/DDP | 4 | BF16 | Profile distributed gradient synchronization | AllReduce time, bandwidth, comm/compute ratio | Must Have | 1.6 |
| E116 | Communication analysis | GPT-2 Small | Training | FSDP/NCCL | 4 | BF16 | Profile sharded communication and memory exchange | Comm overhead, idle time, NVLink utilisation | Must Have | 1.6 |
| E117 | Communication analysis | DistilBERT | Training | NCCL/DDP | 2 | BF16 | Profile distributed gradient synchronization | AllReduce time, bandwidth, comm/compute ratio | Must Have | 0.8 |
| E118 | Communication analysis | DistilBERT | Training | FSDP/NCCL | 2 | BF16 | Profile sharded communication and memory exchange | Comm overhead, idle time, NVLink utilisation | Must Have | 0.8 |
| E119 | Communication analysis | DistilBERT | Training | NCCL/DDP | 4 | BF16 | Profile distributed gradient synchronization | AllReduce time, bandwidth, comm/compute ratio | Must Have | 1.6 |
| E120 | Communication analysis | DistilBERT | Training | FSDP/NCCL | 4 | BF16 | Profile sharded communication and memory exchange | Comm overhead, idle time, NVLink utilisation | Must Have | 1.6 |

## Hardware Comparison

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E121 | Hardware comparison | GPT-2 Small | Training | H100 | 1 | BF16 | Single-GPU H100 baseline comparison | Throughput, memory, power, GPU-hours | Must Have | 0.8 |
| E122 | Hardware comparison | GPT-2 Small | Inference | H100 | 1 | BF16 | Single-GPU H100 inference comparison | Tokens/sec, p95 latency, memory, power | Should Have | 0.5 |
| E123 | Hardware comparison | GPT-2 Small | Training | L40S | 1 | BF16 | Single-GPU L40S baseline comparison | Throughput, memory, power, GPU-hours | Should Have | 0.8 |
| E124 | Hardware comparison | GPT-2 Small | Inference | L40S | 1 | BF16 | Single-GPU L40S inference comparison | Tokens/sec, p95 latency, memory, power | Should Have | 0.5 |
| E125 | Hardware comparison | DistilBERT | Training | H100 | 1 | BF16 | Single-GPU H100 baseline comparison | Throughput, memory, power, GPU-hours | Must Have | 0.8 |
| E126 | Hardware comparison | DistilBERT | Inference | H100 | 1 | BF16 | Single-GPU H100 inference comparison | Tokens/sec, p95 latency, memory, power | Should Have | 0.5 |
| E127 | Hardware comparison | DistilBERT | Training | L40S | 1 | BF16 | Single-GPU L40S baseline comparison | Throughput, memory, power, GPU-hours | Should Have | 0.8 |
| E128 | Hardware comparison | DistilBERT | Inference | L40S | 1 | BF16 | Single-GPU L40S inference comparison | Tokens/sec, p95 latency, memory, power | Should Have | 0.5 |

## Inference Framework

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E129 | Inference framework | GPT-2 Small | Inference | PyTorch eager | 1 | BF16 | Compare PyTorch eager inference engine | Tokens/sec, TTFT, p50, p95, memory | Must Have | 0.6 |
| E130 | Inference framework | GPT-2 Small | Inference | vLLM | 1 | BF16 | Compare vLLM inference engine | Tokens/sec, TTFT, p50, p95, memory | Must Have | 0.6 |
| E131 | Inference framework | GPT-2 Small | Inference | TensorRT-LLM | 1 | BF16 | Compare TensorRT-LLM inference engine | Tokens/sec, TTFT, p50, p95, memory | Should Have | 0.6 |
| E153 | Inference framework | DistilBERT | Inference | PyTorch eager | 1 | BF16 | Compare PyTorch eager inference engine | Tokens/sec, TTFT, p50, p95, memory | Must Have | 0.6 |
| E154 | Inference framework | DistilBERT | Inference | vLLM | 1 | BF16 | Compare vLLM inference engine | Tokens/sec, TTFT, p50, p95, memory | Must Have | 0.6 |
| E155 | Inference framework | DistilBERT | Inference | TensorRT-LLM | 1 | BF16 | Compare TensorRT-LLM inference engine | Tokens/sec, TTFT, p50, p95, memory | Should Have | 0.6 |
| E177 | Inference framework | TinyLlama | Inference | PyTorch eager | 1 | BF16 | Compare PyTorch eager inference engine | Tokens/sec, TTFT, p50, p95, memory | Must Have | 0.8 |
| E178 | Inference framework | TinyLlama | Inference | vLLM | 1 | BF16 | Compare vLLM inference engine | Tokens/sec, TTFT, p50, p95, memory | Must Have | 0.8 |
| E179 | Inference framework | TinyLlama | Inference | TensorRT-LLM | 1 | BF16 | Compare TensorRT-LLM inference engine | Tokens/sec, TTFT, p50, p95, memory | Should Have | 0.8 |

## Inference Batching

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E132 | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Serving batch size 1 | Tokens/sec, latency, memory, saturation point | Must Have | 0.4 |
| E133 | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Serving batch size 2 | Tokens/sec, latency, memory, saturation point | Should Have | 0.4 |
| E134 | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Serving batch size 4 | Tokens/sec, latency, memory, saturation point | Must Have | 0.4 |
| E135 | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Serving batch size 8 | Tokens/sec, latency, memory, saturation point | Should Have | 0.4 |
| E136 | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Serving batch size 16 | Tokens/sec, latency, memory, saturation point | Must Have | 0.4 |
| E137 | Inference batching | GPT-2 Small | Inference | vLLM | 1 | BF16 | Serving batch size 32 | Tokens/sec, latency, memory, saturation point | Should Have | 0.4 |
| E156 | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Serving batch size 1 | Tokens/sec, latency, memory, saturation point | Must Have | 0.4 |
| E157 | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Serving batch size 2 | Tokens/sec, latency, memory, saturation point | Should Have | 0.4 |
| E158 | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Serving batch size 4 | Tokens/sec, latency, memory, saturation point | Must Have | 0.4 |
| E159 | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Serving batch size 8 | Tokens/sec, latency, memory, saturation point | Should Have | 0.4 |
| E160 | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Serving batch size 16 | Tokens/sec, latency, memory, saturation point | Must Have | 0.4 |
| E161 | Inference batching | DistilBERT | Inference | vLLM | 1 | BF16 | Serving batch size 32 | Tokens/sec, latency, memory, saturation point | Should Have | 0.4 |
| E180 | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Serving batch size 1 | Tokens/sec, latency, memory, saturation point | Must Have | 0.5 |
| E181 | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Serving batch size 2 | Tokens/sec, latency, memory, saturation point | Should Have | 0.5 |
| E182 | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Serving batch size 4 | Tokens/sec, latency, memory, saturation point | Must Have | 0.5 |
| E183 | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Serving batch size 8 | Tokens/sec, latency, memory, saturation point | Should Have | 0.5 |
| E184 | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Serving batch size 16 | Tokens/sec, latency, memory, saturation point | Must Have | 0.5 |
| E185 | Inference batching | TinyLlama | Inference | vLLM | 1 | BF16 | Serving batch size 32 | Tokens/sec, latency, memory, saturation point | Should Have | 0.5 |

## Inference Concurrency

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E138 | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Concurrent requests 1 | Throughput, p50, p95, queueing, memory | Must Have | 0.4 |
| E139 | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Concurrent requests 2 | Throughput, p50, p95, queueing, memory | Should Have | 0.4 |
| E140 | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Concurrent requests 4 | Throughput, p50, p95, queueing, memory | Must Have | 0.4 |
| E141 | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Concurrent requests 8 | Throughput, p50, p95, queueing, memory | Must Have | 0.4 |
| E142 | Inference concurrency | GPT-2 Small | Inference | vLLM | 1 | BF16 | Concurrent requests 16 | Throughput, p50, p95, queueing, memory | Should Have | 0.4 |
| E162 | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Concurrent requests 1 | Throughput, p50, p95, queueing, memory | Must Have | 0.4 |
| E163 | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Concurrent requests 2 | Throughput, p50, p95, queueing, memory | Should Have | 0.4 |
| E164 | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Concurrent requests 4 | Throughput, p50, p95, queueing, memory | Must Have | 0.4 |
| E165 | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Concurrent requests 8 | Throughput, p50, p95, queueing, memory | Must Have | 0.4 |
| E166 | Inference concurrency | DistilBERT | Inference | vLLM | 1 | BF16 | Concurrent requests 16 | Throughput, p50, p95, queueing, memory | Should Have | 0.4 |
| E186 | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Concurrent requests 1 | Throughput, p50, p95, queueing, memory | Must Have | 0.5 |
| E187 | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Concurrent requests 2 | Throughput, p50, p95, queueing, memory | Should Have | 0.5 |
| E188 | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Concurrent requests 4 | Throughput, p50, p95, queueing, memory | Must Have | 0.5 |
| E189 | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Concurrent requests 8 | Throughput, p50, p95, queueing, memory | Must Have | 0.5 |
| E190 | Inference concurrency | TinyLlama | Inference | vLLM | 1 | BF16 | Concurrent requests 16 | Throughput, p50, p95, queueing, memory | Should Have | 0.5 |

## Inference Prompt Length

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E143 | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Prompt length 128 tokens | TTFT, generation rate, memory, p95 latency | Must Have | 0.4 |
| E144 | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Prompt length 512 tokens | TTFT, generation rate, memory, p95 latency | Should Have | 0.4 |
| E145 | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Prompt length 1024 tokens | TTFT, generation rate, memory, p95 latency | Must Have | 0.4 |
| E146 | Inference prompt length | GPT-2 Small | Inference | vLLM | 1 | BF16 | Prompt length 2048 tokens | TTFT, generation rate, memory, p95 latency | Should Have | 0.4 |
| E167 | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Prompt length 128 tokens | TTFT, generation rate, memory, p95 latency | Must Have | 0.4 |
| E168 | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Prompt length 512 tokens | TTFT, generation rate, memory, p95 latency | Should Have | 0.4 |
| E169 | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Prompt length 1024 tokens | TTFT, generation rate, memory, p95 latency | Must Have | 0.4 |
| E170 | Inference prompt length | DistilBERT | Inference | vLLM | 1 | BF16 | Prompt length 2048 tokens | TTFT, generation rate, memory, p95 latency | Should Have | 0.4 |
| E191 | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Prompt length 128 tokens | TTFT, generation rate, memory, p95 latency | Must Have | 0.5 |
| E192 | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Prompt length 512 tokens | TTFT, generation rate, memory, p95 latency | Should Have | 0.5 |
| E193 | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Prompt length 1024 tokens | TTFT, generation rate, memory, p95 latency | Must Have | 0.5 |
| E194 | Inference prompt length | TinyLlama | Inference | vLLM | 1 | BF16 | Prompt length 2048 tokens | TTFT, generation rate, memory, p95 latency | Should Have | 0.5 |

## Inference Multi-GPU

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E147 | Inference multi-GPU | GPT-2 Small | Inference | vLLM tensor parallel | 2 | BF16 | Multi-GPU inference serving on 2 GPUs | Scaling, tokens/sec, latency, memory | Should Have | 0.8 |
| E148 | Inference multi-GPU | GPT-2 Small | Inference | vLLM tensor parallel | 4 | BF16 | Multi-GPU inference serving on 4 GPUs | Scaling, tokens/sec, latency, memory | Should Have | 1.6 |
| E171 | Inference multi-GPU | DistilBERT | Inference | vLLM tensor parallel | 2 | BF16 | Multi-GPU inference serving on 2 GPUs | Scaling, tokens/sec, latency, memory | Should Have | 0.8 |
| E172 | Inference multi-GPU | DistilBERT | Inference | vLLM tensor parallel | 4 | BF16 | Multi-GPU inference serving on 4 GPUs | Scaling, tokens/sec, latency, memory | Should Have | 1.6 |
| E195 | Inference multi-GPU | TinyLlama | Inference | vLLM tensor parallel | 2 | BF16 | Multi-GPU inference serving on 2 GPUs | Scaling, tokens/sec, latency, memory | Should Have | 0.8 |
| E196 | Inference multi-GPU | TinyLlama | Inference | vLLM tensor parallel | 4 | BF16 | Multi-GPU inference serving on 4 GPUs | Scaling, tokens/sec, latency, memory | Should Have | 1.6 |

## Inference Precision/Quantization

| ID | Phase | Workload | Mode | Stack/Strategy | GPUs | Precision/Config | Objective | Key metrics | Priority | GPUh |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E149 | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | FP16 | Inference using FP16 | Latency, tokens/sec, memory, quality sanity check | Must Have | 0.5 |
| E150 | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | BF16 | Inference using BF16 | Latency, tokens/sec, memory, quality sanity check | Must Have | 0.5 |
| E151 | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | FP8 | Inference using FP8 | Latency, tokens/sec, memory, quality sanity check | Should Have | 0.5 |
| E152 | Inference precision/quantization | GPT-2 Small | Inference | vLLM or TensorRT-LLM | 1 | INT8/4-bit quantized | Inference using INT8/4-bit quantized | Latency, tokens/sec, memory, quality sanity check | Optional | 0.5 |
| E173 | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | FP16 | Inference using FP16 | Latency, tokens/sec, memory, quality sanity check | Must Have | 0.5 |
| E174 | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | BF16 | Inference using BF16 | Latency, tokens/sec, memory, quality sanity check | Must Have | 0.5 |
| E175 | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | FP8 | Inference using FP8 | Latency, tokens/sec, memory, quality sanity check | Should Have | 0.5 |
| E176 | Inference precision/quantization | DistilBERT | Inference | vLLM or TensorRT-LLM | 1 | INT8/4-bit quantized | Inference using INT8/4-bit quantized | Latency, tokens/sec, memory, quality sanity check | Optional | 0.5 |
| E197 | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | FP16 | Inference using FP16 | Latency, tokens/sec, memory, quality sanity check | Must Have | 0.5 |
| E198 | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | BF16 | Inference using BF16 | Latency, tokens/sec, memory, quality sanity check | Must Have | 0.5 |
| E199 | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | FP8 | Inference using FP8 | Latency, tokens/sec, memory, quality sanity check | Should Have | 0.5 |
| E200 | Inference precision/quantization | TinyLlama | Inference | vLLM or TensorRT-LLM | 1 | INT8/4-bit quantized | Inference using INT8/4-bit quantized | Latency, tokens/sec, memory, quality sanity check | Optional | 0.5 |

## Integrated Analysis Tasks

| ID | Analysis task | Purpose | Notes |
| --- | --- | --- | --- |
| A01 | Training efficiency comparison | Identify best training setup by workload and GPU budget | Uses completed experiment results; no additional GPU time |
| A02 | Inference efficiency comparison | Identify best serving setup by latency and throughput target | Uses completed experiment results; no additional GPU time |
| A03 | Bottleneck classification | Classify each workload as compute, memory, communication, I/O, or mixed | Uses completed experiment results; no additional GPU time |
| A04 | GPU-hour comparison | Compare useful work per GPU-hour across configurations | Uses completed experiment results; no additional GPU time |
| A05 | Throughput/Watt comparison | Compare energy efficiency across selected runs | Uses completed experiment results; no additional GPU time |
| A06 | H100 vs L40S synthesis | Summarise hardware trade-offs and when H100 is justified | Uses completed experiment results; no additional GPU time |
| A07 | Dashboard build | Create visual dashboard for scaling, bottlenecks, cost and energy | Uses completed experiment results; no additional GPU time |
| A08 | Deployment recommendations | Translate findings into practical guidance | Uses completed experiment results; no additional GPU time |

## Feasible Reduced Scope

| Group | Included experiments | Estimated GPU-hours |
| --- | --- | --- |
| Environment and baselines | E001-E006 plus GPT-2 and DistilBERT training/inference baselines | 8-10 |
| Workload characterisation | Profiler, memory, data pipeline and inference characterisation for GPT-2 and DistilBERT | 6-8 |
| Scaling | Strong and weak scaling on 1, 2 and 4 H100 GPUs for GPT-2 and DistilBERT | 20-25 |
| Distributed strategy | DDP vs FSDP on 2 and 4 GPUs; selected ZeRO-1/2 if time allows | 14-18 |
| Precision and memory | FP32/FP16/BF16 plus selected FP8; activation checkpointing and batch size sweep | 18-22 |
| Data movement and communication | DataLoader worker sweep, pinned memory, NCCL profiling and communication overhead | 12-15 |
| Inference impact scope | PyTorch vs vLLM, batching, concurrency and prompt length for at least GPT-2 and TinyLlama | 15-20 |
| Analysis/dashboard | Bottleneck classification, GPU-hour efficiency, throughput/Watt and dashboard | 0 |
| Minimum total | Feasible reduced dissertation scope | 95-118 |
