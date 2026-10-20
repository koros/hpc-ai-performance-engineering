# Serving runtime recovery after the Spack relocation

The original private Python launchers reference the inaccessible Jenkins tree.
Use module Python to create a separate venv; the original venv is not modified.
The replacement reuses its packages and editable vLLM source, so keep those inputs
unchanged. This is a recovery overlay, not a portable copy or new package build.

On HPC, from the repository root after loading `slurm/modules_for_hpc_ai`, leave
any active virtualenv, then run:

```bash
python scripts/create_recovered_vllm_env.py
source "$HOME/.local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo-recovered/bin/activate_recovered_vllm"
python -c 'import torch, vllm; print(torch.__version__, vllm.__version__)'
sbatch slurm/smoke/l40s_recovered_model.sbatch
```

The creator refuses an existing destination and downloads nothing. The custom
activation supplies the restored OpenBLAS/MAGMA/CUPTI library paths, removes
module Python package injection, and selects vLLM V0 for request timestamps.
The venv's `.pth` file registers the private packages and editable install even
when the bundle runner clears PYTHONPATH. The normal bundle environment-contract
checks remain enabled; source the custom activation before creating bundles so
the contract probe also receives the library paths.

The two-GPU diagnostic uses offline cached pinned TinyLlama, validates fixed
output counts and finite ordered request timestamps, and writes a JSON report
under `build/runtime_recovery/l40s/model-job-JOBID/REPORT.json`. It is not benchmark
or platform-canary evidence. A missing cache causes failure rather than network
download. Check Slurm exit code, report and stderr. Full platform canaries and
H100 execution remain separate validation steps.

After diagnostic success, select this activation using `--venv-activate` or
`VENV_ACTIVATE` for regular submission. Do not bypass the existing canary gate.
The underlying Spack libraries changed location and were rebuilt: retain the
recovery manifest, diagnostics and new environment provenance alongside results;
matching package versions alone does not establish measurement equivalence.
