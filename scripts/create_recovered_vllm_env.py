#!/usr/bin/env python3
"""Create a separate submission-compatible venv reusing private serving packages."""

import argparse
import json
import os
import shlex
import sys
import sysconfig
import venv
from pathlib import Path


def create_environment(destination, private, libraries):
    destination = destination.resolve()
    private = private.resolve()
    packages = private / "lib/python3.11/site-packages"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing environment: {destination}")
    for path in [packages, *libraries]:
        if not path.is_dir():
            raise FileNotFoundError(path)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(destination)
    # Process editable-install .pth registrations at every interpreter startup,
    # including vLLM and multiprocessing children with PYTHONPATH cleared.
    purelib = Path(sysconfig.get_path("purelib", vars={"base": str(destination), "platbase": str(destination)}))
    (purelib / "recovered_private_packages.pth").write_text(
        f"import sys, site; sys.path.insert(0, {str(packages)!r}); site.addsitedir({str(packages)!r})\n"
    )
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p and "stubs" not in Path(p).parts]
    library_path = ":".join(map(str, [*libraries, *existing]))
    activation = destination / "bin/activate_recovered_vllm"
    activation.write_text(
        "# Source after slurm/modules_for_hpc_ai; private package tree must remain unchanged.\n"
        f"source {shlex.quote(str(destination / 'bin/activate'))}\n"
        "unset PYTHONHOME PYTHONPATH\n"
        f"export LD_LIBRARY_PATH={shlex.quote(library_path)}\n"
        "export VLLM_USE_V1=0\n"
    )
    (destination / "RECOVERY_MANIFEST.json").write_text(
        json.dumps(
            {
                "base_python": sys._base_executable,
                "private_packages": str(packages),
                "library_path": library_path,
                "python_version": sys.version,
                "note": "Packages are reused, not copied. Full model validation required.",
            },
            indent=2,
        )
        + "\n"
    )
    return activation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path.home() / ".local/share/hpc-ai-performance-engineering"
    parser.add_argument("--destination", type=Path, default=root / "vllm-0.10.2-gloo-recovered")
    parser.add_argument("--private-env", type=Path, default=root / "vllm-0.10.2-gloo")
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 11, 11):
        raise SystemExit("Run with restored module Python 3.11.11")
    compiler = Path(sys.base_prefix).parent
    activation = create_environment(
        args.destination,
        args.private_env,
        [
            compiler / "openblas-0.3.30-k7rfcfy55jybtoth4dltypm7bgdc2a7t/lib",
            compiler / "magma-2.9.0-dhm3llc3gewi6ritwlg4ayndmgehfl5p/lib",
            Path("/scratch/software/cuda/12.4.1/toolkit/extras/CUPTI/lib64"),
        ],
    )
    print(f"Created: {activation}\nOriginal environment unchanged. Source this activation before submission.")


if __name__ == "__main__":
    main()
