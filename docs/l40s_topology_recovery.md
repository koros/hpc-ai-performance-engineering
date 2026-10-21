# Recover interrupted L40S topology comparisons

Historical bundle: `l40s-wave-scaling-dd-topology-0afee5e5-20261018t125822z`.
Reported evidence: 40 valid successes and three E617 failures (43 JSON files).
Twenty-three trials have no result JSON. Preserve this bundle and its source;
never edit its manifests, configs, failures or successful records.

## Confirmed failure causes

- E617: all three trials failed because global `NCCL_ALGO=Tree` prevented
  an AllGather required by the training initialization/placement path.
- E626 repeat 2: cuDNN loader failure (`libcudnn_ops_infer.so.8`), followed by SIGABRT.
- E626 repeat 3 and inspected E627 trials: module startup failed to load
  `openmpi/5.0.5-skres45`. Later missing trials need their own logs before
  assigning an exact failure cause. Do not infer their causes from scheduler state alone.

Repository comparison of the four fingerprinted files confirms the `0afee5e`
snapshot fingerprint is `2b3d8e4d0c702d0def2b6ff8defc6eda5f19d106d8946e4c98b043ed2e479bec`;
the current fingerprint is `b11813be2d4548e06969b4f4d5b98218d2725ff5c2c4b10ca2693b5e24abeac1`.
`workloads/transformer.py` changed; the other three files match. The HPC audit
still needs to check fingerprints actually recorded in the historical bundle.

The environment problems and algorithm problem are separate. NCCL 2.24+ supports
per-collective algorithm selection. New configs use `NCCL_ALGO=allreduce:Tree`
and a corresponding `allreduce:Ring` treatment. Other collectives retain automatic
selection. Reference: [NCCL 2.27 environment variables](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2275/user-guide/docs/env.html#nccl-algo).
This is a corrected algorithm experiment, not a successful rerun of the original
invalid global-Tree setting. No runtime/fingerprinted workload file is changed.

## Preserve and audit first (HPC)

Use the recovered environment, from the main repository root:

```bash
source slurm/modules_for_hpc_ai
source "$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo-recovered/bin/activate_recovered_vllm"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
BUNDLE="../hpc-ai-submissions/l40s-wave-scaling-dd-topology-0afee5e5-20261018t125822z"

python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave scaling-dd-topology --allow-incomplete
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
python scripts/audit_topology_recovery.py --bundle "$BUNDLE"
python -m hpc_ai_perf.cli stage-platform-wave \
  --platform-profile platforms/l40s.json --wave scaling-dd-topology
git diff --cached --stat
# Check for unrelated staged L40S evidence before committing.
git commit -m "data: preserve partial historical L40S topology evidence"
```

Skip that commit if already committed and the index has no changes. Partial
collection preserves validated completed records; retain the original bundle
as the source of failed records and scheduler/log evidence as well.
The read-only audit saves `build/topology_recovery/compatibility_audit.json`,
including file hashes, schema-validation issues, historical runtime fingerprints,
PyTorch/CUDA/cuDNN/NCCL/Transformers versions and available recent current-runtime
examples. Its file hashes are observations, not a replacement for canonical
SHA256SUMS verification. It does not certify binary equivalence. The historical
bundle is remote and was unavailable for local execution of this audit.

## Narrow recovery cohort and controls

Historical and recovered measurements stay separate even if version strings
match: build/library-path equivalence has not been established. New IDs avoid
silently combining old and new repeats. Completed GPT-2 placement/P2P comparisons
E608–E615 and E618 are **not** repeated.

`topology-recovery-canary` has five one-trial conditions, 10 warmup + 20 measured
updates. They exercise GPT-2 eight-GPU Tree AllReduce; TinyLlama pinned four-GPU
placement; pinned two-GPU P2P-disabled placement; eight-GPU Tree AllReduce; and
eight-GPU P2P-disabled training. INFO initialization/environment/graph logs are
retained for diagnostic inspection. Performance configs omit INFO logging.

| New canary | Historical source shape |
|---|---|
| E791 | E617: GPT-2 eight-GPU, corrected Tree |
| E792 | E628: TinyLlama four GPUs, devices 0,2,4,6 |
| E793 | E630: TinyLlama two GPUs, devices 0,4, P2P disabled |
| E794 | E632: TinyLlama eight-GPU, corrected Tree |
| E795 | E633: TinyLlama eight-GPU, P2P disabled |

The acceptance gate requires current configuration/runtime/checksums, clean source,
matching library versions including cuDNN/NCCL, NCCL >= 2.24, complete optimizer
and backward accounting, finite losses/throughput, memory below the existing
ceiling, distinct rank GPU UUIDs, captured PCIe/topology data and correct declared
visibility for pinned allocations. It does not prove a particular kernel was
selected: inspect NCCL logs before attributing mechanism or exact algorithm use.

`topology-recovery-performance` is 14 conditions x 3 trials = **42 trials**, each
10 warmup + 200 measured updates. This is deliberately more than the 26 historical
incomplete trials because matched controls and a complete new E626-shaped cohort
are needed. It is substantially narrower than repeating all 66 trials.

| New IDs | Purpose | Historical source shapes |
|---|---|---|
| E796–E798 | GPT-2 eight-GPU automatic/Ring/Tree comparison | E389, E616, E617 |
| E799–E800 | TinyLlama normal P2P controls at devices 0,1 and 0,4 | E623, E625 |
| E801–E803 | TinyLlama four-GPU placement comparison | E626–E628 |
| E804–E805 | TinyLlama matching two-GPU P2P-disabled comparisons | E629–E630 |
| E806–E809 | TinyLlama eight-GPU automatic/Ring/Tree/P2P-disabled | E421, E631–E633 |

Old successful E626 repeat 1 is retained for historical analysis, not pooled with
three new repeats. Controls are explicit new-ID `comparison_control_id` links.
No historical denominator is used for current-runtime scaling efficiency. These
are within-GPU-count topology/settings comparisons, not a complete new weak-scaling
curve. Pinned-subset conditions reserve all eight GPUs; report **reserved GPU-hours**
as well as active-rank work. Planning estimates are conservative; review canary
timings before spending the full performance allocation.

## Submit acceptance only

After pushing/pulling the implementation, with the recovered environment active:

```bash
# Prevent shell settings from contaminating default/control conditions.
unset NCCL_ALGO NCCL_PROTO NCCL_P2P_DISABLE NCCL_DEBUG NCCL_DEBUG_SUBSYS CUDA_VISIBLE_DEVICES
python scripts/generate_topology_recovery.py --check
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave topology-recovery-canary \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

Do not queue this alongside another full-node L40S wave. After it finishes:

```bash
python -m hpc_ai_perf.cli status-platform-wave \
  --platform-profile platforms/l40s.json --wave topology-recovery-canary
python -m hpc_ai_perf.cli collect-platform-wave \
  --platform-profile platforms/l40s.json --wave topology-recovery-canary
python -m hpc_ai_perf.cli verify-result-evidence --result-root results/raw
python scripts/verify_topology_recovery_canaries.py
```

Inspect NCCL INFO logs and the observed rank/topology mapping. Device labels
such as same-switch/cross-socket must come from captured topology, not device
numbers alone. Kernel algorithm attribution remains unconfirmed until logs or
profiling establish it. Then, if acceptance is sound and timing fits the budget:

```bash
python -m hpc_ai_perf.cli submit-platform-wave \
  --platform-profile platforms/l40s.json --wave topology-recovery-performance \
  --venv-activate "$VIRTUAL_ENV/bin/activate_recovered_vllm"
```

Submission enforces the acceptance gate before creating/submitting the performance
bundle. Historical `scaling-dd-topology` remains incomplete by its original policy;
the new wave is a separate corrected recovery, not a backfill that hides failures.
