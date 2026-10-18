# H100 NVLink capability probe

This is the first step toward NVLink instrumentation, not a performance study or
an acceptance gate for the existing experiment matrix. It checks whether the
allocated H100 GPUs expose readable DCGM GPU-level NVLink TX/RX rates to the
current user. It does not generate transfers, calibrate bandwidth, measure
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

The probe requires the existing NVIDIA driver tools and `dcgmi` in PATH, with
access to the local DCGM host engine/profiling fields. It does not install
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
  two numeric, nonnegative TX and RX samples. Zero is a valid idle reading. This
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
values and implausible/sentinel values are never treated as valid throughput.
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

Fields 1011/1012 are profiling rates (bytes/second), not cumulative counters.
Local field catalogue presence alone does not prove hardware/access support.
Per-link or NVLink5-specific fields are outside this initial H100 probe.
