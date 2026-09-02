# GPU Telemetry Revalidation

## Why this exists

The original `nvidia-smi` collector selected the first *N* rows returned by the command. On this cluster Slurm normally
filters and reindexes visible devices, so the existing evidence is not proven incorrect, but the raw records do not
independently prove that each sampled row belonged to the CUDA devices allocated to the job. Existing raw JSON remains
immutable and is labelled `legacy_positional` only in derived outputs.

The corrected collector resolves CUDA-visible device UUIDs through PyTorch, queries UUIDs with every `nvidia-smi`
sample, and selects rows by UUID. If PyTorch does not expose UUIDs, it accepts the scheduler-filtered fallback only when
`nvidia-smi` returns exactly the requested number of GPUs. It never takes the first *N* rows from a larger set.

New records include:

- `nvidia_smi.gpu_mapping_method`: `cuda_uuid` or `cgroup_visible_rows`;
- `nvidia_smi.profiled_gpu_uuids` and `nvidia_smi.profiled_gpu_count`;
- `nvidia_smi.telemetry_schema_version: 2`;
- metric scopes showing that memory is the maximum per GPU, power is summed across visible GPUs, and utilisation is
  averaged across visible GPUs.

Processed output translates those methods into `uuid_verified` or `cgroup_verified`. Historical records with samples
but no mapping metadata become `legacy_positional`; no raw record is rewritten.

## Audit existing evidence

Run:

```bash
make audit-telemetry
```

This writes `results/analysis/telemetry_audit.csv` and `results/analysis/telemetry_audit.md`. The audit checks mapping
provenance and flags large repeated-trial deviations in GPU utilisation, peak external memory, and average power. A
flag is a review prompt, not an automatic exclusion. Inspect the raw record, Slurm logs, sample count, Torch peak-memory
metric, and workload runtime before deciding whether a measurement failed.

## Revalidation anchors

The same four protocols are materialized independently for L40S and H100. They use the `platform_specific`
comparability class and therefore cannot be pooled with the original experiment families.

| Anchor | Protocol | Reference | GPUs |
| --- | --- | --- | ---: |
| `E210` | GPT-2 BF16 training baseline | `E008` | 1 |
| `E211` | GPT-2 BF16 DDP weak-scaling training | `E036` | 2 |
| `E212` | GPT-2 BF16 PyTorch eager inference | `E129` | 1 |
| `E213` | GPT-2 BF16 DDP/NCCL communication | `E113` | 2 |

Run L40S before starting H100 collection:

```bash
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=telemetry-revalidation
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=telemetry-revalidation
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=telemetry-revalidation
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=telemetry-revalidation
```

Repeat with `PLATFORM=h100` before relying on H100 external telemetry.

## Interpretation policy

Accept an anchor set only when all three trials complete, each record has an explicit mapping method, the profiled UUID
count matches the requested GPU count, and the audit reveals no unexplained mapping or physical-range failure. Compare
UUID-corrected medians and trial variation with their referenced legacy conditions; document the comparison rather
than imposing an arbitrary universal variance cutoff.

If corrected telemetry agrees within the observed run-to-run variation, retain legacy power, utilisation, memory, and
energy measurements with the documented mapping limitation. If a material systematic difference remains, invalidate
only conclusions that depend on the affected external metric and rerun representative conditions in those families.
Throughput, latency, loss, NCCL timings, and independently collected Torch peak-memory values are not invalidated merely
because an old record is labelled `legacy_positional`.
