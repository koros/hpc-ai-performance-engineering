# Dev Experiment Matrix

This document explains the local `dev` profile: why these experiments exist, why they are deliberately smaller than the real HPC matrix, which parameters were chosen, and which experiment families must stay on Lovelace/H100.

The dev profile is not a substitute for dissertation measurements. It is a practical local readiness layer for checking that workload code, model loading, result writing, aggregation, tracking, and dashboard refresh work before GPU budget is spent.

## Why A Dev Profile Exists

The full experiment matrix targets the Lovelace HPC facility and H100/L40S GPUs. When HPC is unavailable, a Mac M1 can still test important software behavior:

- config files load correctly;
- the local runner can execute registered workloads;
- optional PyTorch and Transformers workloads import correctly;
- `device: auto` selects MPS on Apple Silicon, CUDA on GPU nodes, or CPU as fallback;
- raw JSON results are written consistently;
- processed CSVs, analysis tables, trackers, and dashboard inputs refresh correctly;
- small parameter changes affect runtime and throughput in the expected direction.

Those checks give confidence in the experiment pipeline without claiming that Mac M1 numbers represent H100 performance.

## Why The Full Matrix Cannot Run Locally

Most `E001-E200` experiments are intentionally HPC-only because they require hardware, software, or measurement conditions that a local Mac cannot provide.

| Requirement | Why It Is HPC-Only |
| --- | --- |
| H100 or L40S GPU measurements | The dissertation needs performance on target accelerator hardware, not Apple MPS or CPU. |
| Multi-GPU scaling | Strong/weak scaling, DDP, FSDP, ZeRO, tensor parallelism, and communication studies require 2-4 GPUs. |
| NCCL and interconnect metrics | NCCL bandwidth, AllReduce time, synchronisation overhead, and NVLink behavior require NVIDIA GPU nodes. |
| CUDA-specific precision | FP16/BF16 behavior is meaningful on H100; FP8/int8 paths require accelerator/library support not represented by Mac MPS. |
| GPU memory pressure | Activation checkpointing, micro-batch limits, and memory utilisation must be measured against real H100/L40S memory. |
| Inference serving frameworks | vLLM and TensorRT-LLM are GPU-serving paths; their conclusions would not be valid from a Mac run. |
| Cost and GPU-hour accounting | The budget analysis depends on per-GPU HPC execution time, not local wall-clock time. |
| Power/energy metrics | Throughput per Watt and energy measurements depend on cluster tooling and platform counters. |

The dev profile therefore samples only a few safe, local-friendly versions of the larger ideas: baseline training/inference, sequence length, micro-batch size, generated-token length, and synthetic throughput.

## Parameter Choices

The chosen dev parameters are small on purpose. They should run on a Mac M1 with limited memory while still exercising meaningful code paths.

| Parameter | Dev Values | Why These Values Were Chosen |
| --- | --- | --- |
| `device` | `auto` | Lets the same config pick CUDA on GPU nodes, MPS on Mac, or CPU as fallback. |
| `runtime.precision` | `fp32` | Avoids misleading local mixed-precision behavior; H100 precision studies remain HPC-only. |
| `runtime.max_steps` | short runs, usually 5 or 10 | Confirms execution and metrics quickly without turning local checks into long benchmarks. |
| GPT-2 `micro_batch_size` | 1, 2, 4 | Exercises memory/work shape changes while staying realistic for Mac M1 memory. |
| GPT-2 `sequence_length` | 128 | Keeps training cheap while still using real causal-LM token tensors. |
| DistilBERT `sequence_length` | 64, 128, 256 | Checks encoder workload scaling with shorter, baseline, and longer input shapes. |
| Inference `max_new_tokens` | 8, 16, 24, 32, 64 | Checks generation latency/throughput behavior without long local generation runs. |
| TinyLlama `hf_model_name` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | Exercises the TinyLlama-style causal-LM path without downloading/running the full 1.1B model locally. |
| Synthetic `tokens_per_step` | 8192, 32768 | Tests aggregation and throughput math with no PyTorch/Transformers dependency. |
| Synthetic `sleep_seconds` | 0.001 | Produces measurable elapsed time while keeping the run very fast. |

## Dev Experiments

| Dev ID | Workload | Mode | Main Parameter | Dev Value | Related HPC Scope | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| `dev_distilbert_inference_baseline` | DistilBERT | inference | baseline | one sentence, `num_labels=2` | `E014` | Confirms encoder inference, tokenizer/model loading, logits, latency, and result writing. |
| `dev_distilbert_training_baseline` | DistilBERT | training | baseline | batch 2, sequence 128 | `E012` | Confirms encoder training, synthetic labels, loss/backward/optimizer, and throughput. |
| `dev_distilbert_sequence_length_64` | DistilBERT | training | `sequence_length` | 64 | `E024-E026` | Checks shorter encoder input shape. |
| `dev_distilbert_sequence_length_256` | DistilBERT | training | `sequence_length` | 256 | `E024-E026` | Checks longer encoder input shape without full HPC profiler scope. |
| `dev_gpt2_small_inference_baseline` | GPT-2 Small | inference | baseline | 32 generated tokens | `E010` | Confirms decoder inference and generated-token metrics. |
| `dev_gpt2_small_training_baseline` | GPT-2 Small | training | baseline | batch 1, sequence 128 | `E008` | Confirms decoder training path, loss/backward/optimizer, and token throughput. |
| `dev_gpt2_small_inference_tokens_16` | GPT-2 Small | inference | `max_new_tokens` | 16 | `E143-E146` | Checks shorter local generation length. |
| `dev_gpt2_small_inference_tokens_64` | GPT-2 Small | inference | `max_new_tokens` | 64 | `E143-E146` | Checks longer local generation length. |
| `dev_gpt2_small_micro_batch_2` | GPT-2 Small | training | `micro_batch_size` | 2 | `E065-E068` | Checks a moderate local micro-batch increase. |
| `dev_gpt2_small_micro_batch_4` | GPT-2 Small | training | `micro_batch_size` | 4 | `E065-E068` | Checks a larger local micro-batch increase without pushing into HPC-scale memory testing. |
| `dev_synthetic_tokens_8192` | Synthetic | training | `tokens_per_step` | 8192 | local smoke | Confirms fast result/throughput pipeline without ML dependencies. |
| `dev_synthetic_tokens_32768` | Synthetic | training | `tokens_per_step` | 32768 | local smoke | Confirms synthetic throughput scaling and aggregation. |
| `dev_tinyllama_inference_baseline` | TinyLlama-style | inference | baseline | 24 generated tokens | `E018` | Confirms TinyLlama-style causal-LM inference path using a tiny random model. |
| `dev_tinyllama_inference_tokens_8` | TinyLlama-style | inference | `max_new_tokens` | 8 | `E191-E194` | Checks short generated-token inference path. |
| `dev_tinyllama_inference_tokens_32` | TinyLlama-style | inference | `max_new_tokens` | 32 | `E191-E194` | Checks longer generated-token inference path while staying local-friendly. |

## Dev-Capable Vs HPC-Only Matrix

| Experiment Family | Matrix IDs | Dev Coverage | Where To Run | Reason |
| --- | --- | --- | --- | --- |
| Local smoke plumbing | `configs/local/*` | synthetic, GPT-2, DistilBERT, TinyLlama inference smoke | local or HPC smoke | Minimal dependency and workflow checks. |
| Dev baseline and small sweeps | `configs/dev/*` | 15 dev configs | local Mac/workstation | Confirms code paths and small parameter effects. |
| Environment validation | `E001-E007` | no direct dev equivalent | HPC | Requires CUDA, NCCL, FSDP, vLLM, and cluster environment checks. |
| Baseline profiling | `E008-E019` | selected GPT-2, DistilBERT, TinyLlama-style baselines | HPC for dissertation measurements | Dev checks the adapter paths; final metrics need H100/L40S. |
| Workload characterisation | `E020-E031` | DistilBERT sequence-length proxy only | HPC | Real compute/memory/data-pipeline profiling needs GPU counters and profiler tools. |
| Strong/weak scaling | `E032-E046` | none | HPC | Requires multiple GPUs and comparable cluster scheduling. |
| Distributed strategies | `E047-E062` | none | HPC | Requires DDP/FSDP/DeepSpeed multi-process orchestration and NCCL. |
| Memory optimisation | `E063-E082` | GPT-2 micro-batch proxy only | HPC | Dev can check shape changes; actual checkpointing/accumulation memory results need H100. |
| Precision studies | `E083-E096` | none | HPC | FP16/BF16/FP8 comparisons must be measured on NVIDIA accelerators. |
| Data movement | `E097-E112` | none | HPC | DataLoader, pinned memory, and prefetch behavior are platform dependent. |
| Communication analysis | `E113-E120` | none | HPC | Requires NCCL, AllReduce, multi-GPU synchronization, and interconnect metrics. |
| Hardware comparison | `E121-E128` | none | HPC | Requires H100 and L40S platforms. |
| Inference framework | `E129-E137`, `E177-E179` | no framework equivalent | HPC | vLLM/TensorRT-LLM/PyTorch eager comparisons are GPU-serving experiments. |
| Inference batching/concurrency | `E138-E142`, `E180-E190` | none | HPC | Needs realistic serving load and GPU runtime behavior. |
| Inference prompt length | `E143-E146`, `E191-E194` | GPT-2/TinyLlama generated-token proxy | HPC | Dev checks generation length mechanics; final prompt-length results need H100. |
| Inference multi-GPU | `E147-E148`, `E195-E196` | none | HPC | Requires tensor-parallel or multi-GPU serving setup. |
| Inference precision/quantization | `E149-E176`, `E197-E200` | none | HPC | Requires GPU/kernel/framework support for target precision and quantization paths. |

## How To Run

Run all dev experiments and refresh outputs:

```bash
make run-dev-refresh
```

Run only the baseline-shaped subset:

```bash
make run-dev-baselines
python -m hpc_ai_perf.cli refresh
make track-dev-progress
```

Run one dev config directly:

```bash
python -m hpc_ai_perf.cli run --config configs/dev/dev_gpt2_small_training_baseline.json
```

Inspect dev progress:

```bash
open results/processed/dev_experiment_tracker.md
```

## Interpretation Rules

- Use dev results to validate workflow readiness and catch code/config problems.
- Do not use dev results as H100 performance evidence.
- Do not mix `dev_` rows into the main dissertation matrix tracker.
- Treat differences between dev runs as local sanity checks, not publishable scaling conclusions.
- Re-run the relevant HPC matrix experiments after Lovelace access is available.
