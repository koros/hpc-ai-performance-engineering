# H100 NVLink capability probe

This is the first step toward NVLink instrumentation, not a performance study or
an acceptance gate for the existing experiment matrix. It checks whether the
allocated H100 GPUs expose readable NVLink TX/RX telemetry through DCGM or
the NVIDIA driver's direct NVML field API to the current user. It does not generate transfers, calibrate bandwidth, measure
per-link traffic or attribute bytes to requests/collectives.

## Run on the cluster

Use the project root and a committed checkout. Wait for other H100 benchmark
jobs to finish or schedule this diagnostic separately; it needs two H100 GPUs.

```bash
make submit-h100-nvlink-probe
```

The target submits `slurm/smoke/h100_nvlink_probe.sbatch` to `gpu_h100`, requesting
two GPUs, four CPUs and a ten-minute limit. The script sources the existing HPC
modules and project venv, then uses CUDA-visible UUIDs to identify the allocated
GPUs. It reads NVIDIA SMI inventory/topology/link state, DCGM version/discovery,
profiling metric groups and five samples of fields 1011/1012 at one-second
intervals. Sampling is limited to DCGM GPU IDs joined by allocated UUIDs; neither
CUDA ordinals nor NVIDIA SMI indices are assumed to equal DCGM IDs. If identity
cannot be established, sampling stops and the reason is recorded.

The probe first tries `dcgmi` and the local DCGM host engine/profiling fields.
If DCGM is missing, inaccessible or inconclusive, it automatically tries the
installed driver library `libnvidia-ml.so.1` through ctypes. No Python NVML
package or DCGM installation is required for the fallback. It does not install
packages, start a host engine, reset counters, run privileged commands or
pause/resume monitoring. Temporary DCGM watches end when its dmon client exits.
The CUDA identity lookup can initialize CUDA contexts but launches no transfer
or compute benchmark. It does not download models or change GPU settings.

Read results after the job finishes, replacing JOBID with the submitted job ID:

```bash
cat build/nvlink_probe/h100/job-JOBID/SUMMARY.md
cat build/nvlink_probe/h100/job-JOBID/PROBE.json
```

Scheduler logs are `slurm/h100-nvlink-probe-JOBID.out` and `.err`. Output directories
are unique per Slurm job and cannot be overwritten. Output lives under ignored
`build/`, separate from canonical E-number results and checksum manifests.
For a diagnostic inside an existing two-GPU Slurm allocation, the equivalent is:

```bash
PYTHONPATH=src python scripts/probe_h100_nvlink.py \
  --expected-gpus 2 --output-dir build/nvlink_probe/h100/manual-UNIQUE_LABEL
```

## Interpretation

- `readable_requires_calibration` (exit 0): every allocated GPU returned at least
  two numeric, nonnegative TX and RX samples (for NVML, with distinct driver
  timestamps). Zero is a valid idle reading. This
  establishes access to numeric telemetry only, not transfer accuracy or
  performance suitability. Positive traffic is reported without attributing it.
- `blocked_or_inconclusive` (exit 2): missing tools, unknown GPU mapping, failed
  commands, too few samples or unavailable/invalid values. Inspect the raw
  stdout/stderr and return codes in PROBE.json. `N/A` can mean unsupported,
  permission denied, uninitialized or unavailable; the table cannot reliably
  distinguish these. Failure through DCGM does not rule out another supported API.

JSON includes tool paths, commands, command timestamps, timeout/error output,
source commit/status, probe script checksum, GPU identity, driver version,
Slurm job ID, sample coverage and raw sampled rates. NaN, infinity, negative
values and explicit unavailable sentinels are never treated as valid telemetry.
The selected backend appears in `assessment.backend`. Direct NVML results
are under `nvml`, and its bounded subprocess output under `commands.nvml_fallback`.
Original DCGM failures remain visible even when the NVML fallback succeeds.
The fallback resolves handles by allocated UUID, verifies each returned UUID,
and queries fields 138/139 with UINT_MAX scope (aggregate across links). It saves
per-field return codes, error messages, value types, driver timestamps and latency.
Cached readings with identical timestamps do not count as independent samples.
Native calls are bounded by a subprocess timeout. No counters are reset.

The requested sample interval is recorded; this capability probe does not
construct a precisely timestamped benchmark time series or integrate byte totals.

A missing tool should be addressed through the site's available module/service.
A permission or host-engine problem may require the HPC administrator's help;
do not change shared monitoring services to make a user probe pass.

## Next step after readable results

Validate both directions against controlled inter-GPU transfers and a known NCCL
collective, retain an idle baseline, check sample windows and quantify collection
overhead. Only then implement a measured-region collector and supplemental
inference experiments. GPU-level interval rates cannot establish per-link paths,
per-request traffic or communication stall time without additional instrumentation.

No existing workload, canary, runtime fingerprint, result JSON or benchmark
configuration is changed by adding/running this probe. It does not require
rerunning the current campaign's canaries.

References checked 18 October 2026:

- [DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [dcgmi dmon](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgmi/dcgmi-dmon.html)
- [dcgmi discovery](https://docs.nvidia.com/datacenter/dcgm/latest/reference/command-line-reference/dcgmi/dcgmi-discovery.html)

DCGM fields 1011/1012 are profiling rates (bytes/second), not cumulative counters.
NVML fields 138/139 are documented as data throughput in KiB. Their original
units/values are retained separately, without asserting equivalence to DCGM
rates or deriving transferred-byte totals before calibration. NVML field/API
support and permissions still depend on the installed driver and GPU.
Local field catalogue presence alone does not prove hardware/access support.
Per-link or NVLink5-specific fields are outside this initial H100 probe.

Direct NVML references:

- [Field definitions and aggregate scope](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlFieldValueEnums.html)
- [Field sample structure and return codes](https://docs.nvidia.com/deploy/nvml-api/api/structnvmlFieldValue__t.html)

## Diagnosing successful calls with rejected values

A successful per-field return code and fresh timestamp establish API access,
not that its value is usable. The NVML decoder retains `raw_value`, `value_type`,
`raw_union_hex`, timestamp and a specific rejection reason. Non-finite raw values
are strings so the JSON remains standards-compliant. For nonzero per-field API
errors, the union is undefined and is not interpreted.

An earlier probe version discarded values above an assumed 1e13 ceiling. That
ceiling was not justified for the raw NVML KiB fields and has been removed;
large readings are retained without calling them validated bandwidth. Explicit
NVML unavailable values (-1/all-bits-set for the returned type) remain rejected.
Old reports cannot recover rejected values that were not saved; rerun this small
probe to distinguish those cases. Rejection details now appear in SUMMARY.md.

## Controlled-transfer follow-up

After obtaining readable counters and an idle baseline, run the bounded active
diagnostic when the H100 is free of other benchmark work:

```bash
make submit-h100-nvlink-calibration
```

This requests two H100s for at most ten minutes. It allocates one 64 MiB buffer
per GPU, checks bidirectional CUDA peer accessibility, warms up both directions,
records an idle window, then copies for two seconds in each direction. The copy
count and payload size establish known application payload bytes. Both GPU
streams are synchronized around the copy loop and destination contents checked.
Snapshots bracket each phase, include settling/readout time, and preserve the
original NVML readings and timestamps. No counters or GPU settings are reset.
This is a transfer diagnostic, not a model benchmark or an E-number result.

Read `build/nvlink_calibration/h100/job-JOBID/SUMMARY.md` and
`CALIBRATION.json` after completion. Logs use the name
`slurm/h100-nvlink-calibration-JOBID.out` and `.err`. Replace JOBID with the
actual submitted ID.

`responsive_requires_unit_validation` requires a quiet idle baseline, correct
copy contents, advancing counter timestamps, nondecreasing counters and positive
source TX/destination RX deltas in both directions. Otherwise the result is
`inconclusive` and the report retains errors and any partial evidence.

The analysis reports raw delta divided by known application payload bytes for
each endpoint separately. Do not add endpoint TX and RX: that would double-count
the same transfer. Compare the two directions and their ratios to establish
field units and whether the counter behaves cumulatively. A ratio near 1/1024
would be consistent with KiB-counted payload, but is not assumed or used as an
automatic pass criterion. Nonzero background traffic or delayed updates must
be resolved before quantitative claims.

The tool deliberately does not claim calibrated bandwidth, link saturation,
per-link path, or exact physical-wire bytes. CUDA peer access alone does not
prove transport selection; the observed NVLink response is additional evidence.
The copy-loop time excludes readout/settling while the counter interval includes
them, so these durations must not be substituted for each other. Repeat with
several known payload volumes and establish instrumentation overhead before
using these counters in the planned inference study.

## Validate additional payload sizes and preserve evidence

The first 64 MiB test transferred 240 GiB in each direction, with exactly one
raw counter unit per 1,024 payload bytes at both endpoints and zero reverse
traffic. That supports cumulative KiB semantics on the tested H100 pair; keep
the original `job-1793641/CALIBRATION.json` as the 64 MiB evidence.

When no H100 benchmark is running, submit only the two missing payload sizes:

```bash
make submit-h100-nvlink-validation
```

This is a two-task Slurm array with concurrency limited to one. It runs 16 MiB
and 128 MiB buffers using the same diagnostic, with two active seconds in each
direction. It does not repeat the completed 64 MiB test. Each task writes under
`build/nvlink_calibration/h100/job-ARRAYID_TASKID/`. The two tasks must not
overlap other H100 performance measurements; do not submit them concurrently
with the four-GPU inference campaign.

After both tasks finish:

```bash
make verify-h100-nvlink-calibration
```

The verifier reads the latest report for each of 16/64/128 MiB, independently
recomputes deltas from the saved raw snapshots, checks idle/reverse traffic,
driver timestamps, copy correctness, payload accounting and both endpoints.
Converted bytes must match the known payload within 2%. A newer failed report
is not bypassed in favour of an older success. The output manifest records
original report checksums and GPU/runtime metadata; it is scoped to those
recorded setups and does not assert overhead, saturation or per-link attribution.
The original 64 MiB report predates driver-inventory capture: its absent
driver metadata remains explicitly unknown rather than being invented.

Preserve this diagnostic evidence separately from canonical E-number trials:

```bash
mkdir -p results/diagnostics/nvlink/h100
cp -R build/nvlink_calibration/h100/job-* results/diagnostics/nvlink/h100/
cp build/nvlink_calibration/h100/VALIDATED_CALIBRATION.json results/diagnostics/nvlink/h100/
```

Review and commit the JSON reports and manifest through the usual Git workflow.
The manifest retains original cluster paths plus content checksums; archive
copies should be matched by checksum, since paths differ between machines.
No existing experiment config, serving fingerprint or training fingerprint is
changed by these diagnostic additions.

## Productive H100 work while collector development continues

Run the existing unprofiled `inference-crossover-canary` wave on H100 (E719–E721,
three single trials) using the certified private vLLM environment. Collect and
verify it, then run `inference-crossover-short` (18 trials), followed by
`inference-crossover-long` (18 trials), collecting between waves. These trials
provide layout/latency/throughput evidence without NVLink measurements. They
remain separate from later instrumented overhead controls and traffic trials.
Do not run both the legacy E702–E714 campaign and the new crossover grid
automatically; prefer the new equal-budget grid for this study.

The next implementation stage is an opt-in measured-region NVLink collector
with its own provenance and separate instrumented configurations. It must use
validated cumulative-counter deltas and preserve counter-window duration
separately from workload time. An overhead-on/off contrast is required before
using instrumented timing for performance claims. The collector is not yet
implemented or enabled by this validation update.
