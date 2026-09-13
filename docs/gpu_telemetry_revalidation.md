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

Telemetry schema version 3 additionally timestamps samples and separates `measured_region` energy from the broader
collector window. The original E210-E213 L40S runs validated device attribution, but their unqualified average-power
and energy estimates include warm-up and profiler finalization. Preserve those records unchanged. Their throughput,
latency, NCCL, loss, Torch memory, and peak-power evidence remains usable; their energy values are
`legacy_full_window` diagnostics and must not feed tokens/kWh or steady-state power conclusions.

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
| `E214` | Measured-region GPT-2 BF16 DDP/NCCL energy | `E113`, `E213` | 2 |

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

For energy revalidation, use the new immutable `E214` condition rather than rerunning E210-E213 with duplicate trial
identities. E214 is the two-GPU communication protocol corresponding to E113/E213. It
must produce three trials with `telemetry_schema_version=3`, `energy_scope=measured_region`, and
`measurement_status=sufficient`. Compare its throughput and NCCL metrics against E113/E213 for protocol consistency,
then use only the corrected condition for average power, joules, and tokens/kWh. Run the corrected collector before
the first H100 energy campaign; do not copy L40S energy conclusions to H100.

```bash
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=energy-revalidation
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=energy-revalidation
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=energy-revalidation
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=energy-revalidation
```

Repeat with `PLATFORM=h100` before making cross-platform communication-energy claims.

## L40S hardware-comparison energy anchors

The L40S hardware-comparison rows E123/E124/E127/E128 were collected before measured-region energy became the
canonical protocol. Preserve those raw records: their performance evidence remains valid, while their
`legacy_full_window` energy must remain excluded from `hardware_summary.avg_energy_joules`.

Use unique supplemental IDs so the revalidation cannot collide with existing immutable trial identities:

| Anchor | Exact protocol reference | Workload and mode | GPUs |
| --- | --- | --- | ---: |
| `E235` | `E123` | GPT-2 Small training | 1 |
| `E236` | `E124` | GPT-2 Small inference | 1 |
| `E237` | `E127` | DistilBERT training | 1 |
| `E238` | `E128` | DistilBERT inference | 1 |

Each anchor records `energy_reference_experiment_id` and otherwise preserves the reference row's protocol fields.
The profiler interval is 0.05 seconds so short inference regions can still produce multiple timestamped samples.
The hardware summary accepts anchor energy only when the explicit reference and protocol signature both match; it
never pools anchor throughput, latency, or memory into the source row.

```bash
make submit-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
make status-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
make collect-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
make verify-hardware-energy-revalidation PLATFORM=l40s
make stage-platform-wave PLATFORM=l40s PLATFORM_WAVE=hardware-energy-revalidation
```

Do not treat scheduler completion alone as acceptance. The dedicated verifier requires all three repetitions for all
four anchors, schema-v3 `measured_region` energy, `sufficient` status, at least two samples, positive finite joules and
power, an allocation-bound mapping, and the observed L40S model. If a short inference anchor fails that gate, retain
the record for diagnosis and adjust the measurement duration through another uniquely identified protocol rather
than filling or estimating the missing value.
