#!/usr/bin/env python3
"""Run a Python script with restored Spack libraries and existing private packages."""

import os
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/run_recovered_vllm.py SCRIPT [ARGS...]")
    root = Path(sys.base_prefix).parent
    libraries = [
        root / "openblas-0.3.30-k7rfcfy55jybtoth4dltypm7bgdc2a7t/lib",
        root / "magma-2.9.0-dhm3llc3gewi6ritwlg4ayndmgehfl5p/lib",
        Path("/scratch/software/cuda/12.4.1/toolkit/extras/CUPTI/lib64"),
    ]
    private = Path(
        os.environ.get(
            "HPC_RECOVERY_VLLM_ENV", str(Path.home() / ".local/share/hpc-ai-performance-engineering/vllm-0.10.2-gloo")
        )
    )
    packages = private / "lib/python3.11/site-packages"
    for path in [*libraries, packages, Path(sys.argv[1])]:
        if not path.exists():
            raise SystemExit(f"Recovery input missing: {path}")
    env = os.environ.copy()
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "stubs" not in Path(p).parts]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libraries + existing))
    env["HPC_RECOVERY_SITE"] = str(packages)
    env["PYTHONPATH"] = str(packages) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("VIRTUAL_ENV", None)
    bootstrap = """
import os, runpy, site, sys
site.addsitedir(os.environ['HPC_RECOVERY_SITE'])
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    os.execve(sys._base_executable, [sys._base_executable, "-c", bootstrap, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
