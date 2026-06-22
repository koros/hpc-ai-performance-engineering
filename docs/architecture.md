# Project Architecture

This repository is organised as a reproducible experiment platform rather than a collection of one-off benchmark scripts.

## Design Principles

- Workloads are implemented behind a common interface.
- Experiment definitions are configuration driven.
- Runners handle execution policy, not model logic.
- Profilers collect measurements through reusable adapters.
- Results use one shared schema across training, inference, scaling, and profiling experiments.
- Slurm integration is isolated from local execution so experiments can be debugged before HPC submission.

## Top-Level Layout

```text
configs/        Experiment configuration files.
experiments/    Experiment group entry points and orchestration scripts.
src/            Reusable Python framework.
workloads/      Workload-specific adapters and assets.
slurm/          Slurm templates and job scripts.
results/        Raw outputs, processed summaries, and figures.
schemas/        Machine-readable schemas for result validation.
dashboard/      Visualisation application.
notebooks/      Exploratory analysis.
```

## Execution Flow

1. Load an experiment configuration.
2. Resolve the requested workload from the workload registry.
3. Start configured profilers.
4. Execute the workload through a runner.
5. Stop profilers and merge profiling metrics.
6. Write a schema-compatible JSON result.

## Current Scope

The first executable path uses a synthetic workload to validate configuration, result writing, and local execution without requiring GPUs. Concrete NanoGPT, GPT-2, and Llama 3.2 adapters should be added next.

