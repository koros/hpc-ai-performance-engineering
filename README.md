# Performance Engineering of Transformer Training and Inference on Multi-GPU H100 Systems

This repository contains the research artefacts, experiment framework, tracking documents, analysis outputs, and dashboard work for an MSc Artificial Intelligence dissertation at the University of Plymouth.

The project studies how transformer workloads behave on NVIDIA H100 systems in the Lovelace HPC facility, with selected NVIDIA L40S comparisons. Training is the core dissertation scope; inference is retained as a secondary high-impact scope because serving cost and latency often dominate long-term deployment.

## Current Experiment Plan

The latest plan is based on `Appendix_A_Comprehensive_Experiment_Matrix.docx` and uses a controlled factorial design across workload, GPU count, distributed strategy, precision, memory settings, data movement, inference framework, serving load, and hardware.

Primary planning documents:

- [Comprehensive experiment matrix](docs/experiment_matrix.md): full source-of-truth matrix with `E001` to `E200`, priority levels, metrics, and GPU-hour estimates.
- [Experiment tracker](docs/experiment_tracker.md): editable run-status table for every planned experiment and analysis task.
- [Experiment workspace](experiments/README.md): phase folders aligned with the dissertation proposal outline.
- [Experiment configs](configs/README.md): reusable config templates and concrete run configs.

## Proposal Outline

The repository now mirrors the project proposal structure:

| Proposal section | Matrix coverage |
| --- | --- |
| Environment validation | `E001-E007` |
| Benchmark selection | GPT-2 Small, DistilBERT, TinyLlama |
| Baseline profiling | `E008-E019` |
| Precision studies | `E083-E096`, selected inference precision in `E149-E200` |
| Scaling studies | `E032-E046` strong scaling and `E035-E043` weak scaling |
| Distributed training strategies | `E047-E062` DDP, FSDP, ZeRO-1, ZeRO-2, ZeRO-3 |
| Memory optimisation | `E063-E082` |
| Data movement and communication | `E097-E120` |
| Hardware comparison | `E121-E128` H100 vs L40S |
| Inference studies | `E129-E200`, subject to time and platform stability |
| Integrated analysis and dashboard | `A01-A07` |
| Recommendations | `A08` |

## Research Questions

1. How efficiently do transformer workloads scale from one to four H100 GPUs?
2. Which distributed training strategy provides the best throughput-to-memory trade-off?
3. What impact do FP32, FP16, BF16, FP8, and selected quantized inference formats have on performance and quality?
4. How significant are communication, synchronisation, and data movement bottlenecks?
5. Which configurations produce the best useful work per GPU-hour and per Watt?
6. How do H100 and L40S platforms compare for selected AI workloads?
7. Can experiment results be integrated into a dashboard that identifies bottlenecks and supports practical recommendations?

## Workloads

| Workload | Type | Primary use |
| --- | --- | --- |
| GPT-2 Small | Decoder transformer | Training and inference baseline |
| DistilBERT | Encoder transformer | Training and inference comparison |
| TinyLlama | Small modern LLM | Primarily inference; optional training |

## Priority and Budget

| Priority | Meaning | Estimated GPU-hours |
| --- | --- | --- |
| Must Have | Required for a complete dissertation | 89.4 |
| Should Have | Important for stronger impact | 46.1 |
| Optional | Stretch goal | 12.1 |
| Total | Full planned matrix | 147.6 |

The reduced feasible scope in the appendix is approximately 95-118 GPU-hours, prioritising environment validation, GPT-2 and DistilBERT baselines, scaling, distributed strategy comparison, precision and memory studies, data movement, communication analysis, and a focused inference subset.

## Repository Structure

```text
hpc-ai-performance-engineering/
├── configs/                  # JSON/YAML experiment configurations
│   ├── distributed/
│   ├── inference/
│   ├── local/
│   ├── profiling/
│   └── training/
├── docs/                     # Project plan, matrix, tracker, architecture, hardware notes
├── experiments/              # Phase folders aligned with the proposal outline
├── results/                  # Raw results, processed outputs, and figures
├── schemas/                  # Result schema definitions
├── scripts/                  # Utility scripts
├── slurm/                    # Slurm submission templates
├── src/hpc_ai_perf/          # Experiment config, runner, profiling, results, workload code
├── tests/                    # Local validation tests
└── workloads/                # Workload-specific assets and notes
```

## Metrics

The matrix tracks runtime, samples/sec, tokens/sec, loss, GPU-hours, GPU utilisation, GPU memory, CPU utilisation, data-loading time, speedup, scaling efficiency, NCCL bandwidth, AllReduce time, synchronisation overhead, NVLink utilisation, tokens/sec, time-to-first-token, p50 and p95 latency, power draw, energy consumption, throughput/GPU-hour, and throughput/Watt.

## Local Validation

Install the package in editable mode and run the smoke tests:

```bash
pip install -e ".[dev]"
pytest
```

Run the local synthetic smoke benchmark:

```bash
python -m hpc_ai_perf.cli run --config configs/local/synthetic_smoke.json
```

List matrix experiment configs:

```bash
python -m hpc_ai_perf.cli list-configs
```

## Expected Outputs

- Reproducible experiment suite and configuration catalogue
- Baseline, scaling, precision, memory, communication, hardware, and inference results
- Integrated analysis and dashboard
- Dissertation figures and tables
- Practical recommendations for efficient transformer training and serving on modern GPU systems

## Author

Geoffrey Koros
MSc Artificial Intelligence
University of Plymouth

## License

To be determined.
