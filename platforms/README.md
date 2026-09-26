# Platform Execution Policy

The files in this directory turn the logical source matrix into separately identifiable L40S and H100
conditions. They are the source of truth for platform routing; generated configs under `build/` are disposable.

## Files

| File | Purpose |
| --- | --- |
| `l40s.json` | L40S hardware, `gpu_l40s` partition, 8 CPUs/GPU, PCIe, result root, and Slurm profile. |
| `h100.json` | H100 hardware, `gpu_h100` partition, 16 CPUs/GPU, NVLink, result root, and Slurm profile. |
| `execution_matrix.json` | Planned/deferred conditions, execution waves, required trials, and comparability. |

Every logical row appears exactly once for each platform, either planned or deferred. Tests enforce full coverage of
all 200 IDs and reject duplicate conditions.

## Identity and Evidence Separation

The source experiment ID remains `E123`, while the executable condition is `E123@l40s` or `E123@h100`. A generated
config embeds:

- the execution platform and condition ID;
- the source config path and SHA-256;
- wave and required independent-trial count;
- accepted evidence root (`results/raw/l40s` or `results/raw/h100`); and
- `exact`, `capacity_adjusted`, or `platform_specific` comparability.

Do not edit a generated config and treat it as an existing condition. If memory limits force a workload-shape change,
register that policy explicitly as `capacity_adjusted` before collection.

## Commands

```bash
# Current L40S path
make render-l40s-wave PLATFORM_WAVE=0
make track-l40s-progress

# Same matrix when H100 returns
make render-h100-wave PLATFORM_WAVE=0
make track-h100-progress
```

Use [the platform execution runbook](../docs/platform_execution_runbook.md) for gating, submission, and result import.

The `lowbit-serving` (E239–E244), optional `lowbit-bnb` (E245–E246), and `vllm-long-context` (E247–E249) waves each use one GPU per trial, three trials per condition, on both platforms. Use their dedicated `submit-lowbit-wave` or `submit-long-context-wave` wrapper after same-platform acceptance rather than the generic submission command. These are fresh controlled trials, not replacements for legacy FP8, deferred INT8 evidence, or the smaller-model prompt sweep. See the [low-bit runbook](../docs/lowbit_serving_runbook.md) and [long-context runbook](../docs/long_context_serving_runbook.md).
