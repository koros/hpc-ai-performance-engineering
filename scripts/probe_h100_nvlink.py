"""Read-only, allocation-scoped DCGM NVLink capability probe; no benchmark load."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

FIELDS = {"1011": "nvlink_tx_bytes_per_second", "1012": "nvlink_rx_bytes_per_second"}
UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F-]{36}\b")


def run_command(command: list[str], timeout: float = 20) -> dict:
    started = datetime.now(UTC).isoformat()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": command,
            "started_at": started,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": "finished",
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "started_at": started,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "status": "missing_executable",
        }
    except subprocess.TimeoutExpired as exc:

        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        return {
            "command": command,
            "started_at": started,
            "returncode": None,
            "stdout": decode(exc.stdout),
            "stderr": decode(exc.stderr),
            "status": "timeout",
        }
    except OSError as exc:
        return {
            "command": command,
            "started_at": started,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "status": "execution_error",
        }


def dcgm_uuid_mapping(output: str) -> dict[str, int]:
    """Join DCGM's discovery table by UUID, never by CUDA/nvidia-smi ordinal."""
    mapping = {}
    current = None
    for line in output.splitlines():
        match = re.match(r"\s*\|\s*(\d+)\s*\|", line)
        if match:
            current = int(match.group(1))
        uuid = UUID_PATTERN.search(line)
        if current is not None and uuid:
            key = uuid.group().lower()
            if key in mapping and mapping[key] != current:
                raise ValueError("DCGM discovery contains conflicting GPU UUID mappings")
            mapping[key] = current
    return mapping


def parse_dmon(output: str, gpu_ids: set[int]) -> list[dict]:
    """Parse only explicit GPU rows; unavailable/invalid values stay null."""
    samples = []
    for line in output.splitlines():
        columns = line.split()
        if len(columns) != 4 or columns[0] != "GPU" or not columns[1].isdigit():
            continue
        gpu_id = int(columns[1])
        if gpu_id not in gpu_ids:
            continue
        sample = {"gpu_id": gpu_id}
        for field, raw in zip(FIELDS.values(), columns[2:], strict=True):
            try:
                value = float(raw)
            except ValueError:
                value = math.nan
            sample[field] = value if math.isfinite(value) and 0 <= value < 1e13 else None
        samples.append(sample)
    return samples


def assess_samples(samples: list[dict], gpu_ids: set[int], *, command_ok: bool) -> dict:
    coverage = {}
    for gpu in sorted(gpu_ids):
        rows = [s for s in samples if s["gpu_id"] == gpu]
        coverage[str(gpu)] = {field: sum(r[field] is not None for r in rows) for field in FIELDS.values()}
    readable = (
        bool(gpu_ids) and command_ok and all(count >= 2 for counts in coverage.values() for count in counts.values())
    )
    positive = any(s[field] is not None and s[field] > 0 for s in samples for field in FIELDS.values())
    return {
        "status": "readable_requires_calibration" if readable else "blocked_or_inconclusive",
        "valid_samples_by_gpu": coverage,
        "positive_traffic_observed": positive,
        "traffic_attribution_validated": False,
        "bandwidth_calibrated": False,
    }


def allocated_gpu_uuids(expected: int) -> tuple[str, ...]:
    from hpc_ai_perf.profiling.system import visible_cuda_device_uuids

    uuids = visible_cuda_device_uuids(expected)
    if len(uuids) != expected or len(set(u.lower() for u in uuids)) != expected:
        raise ValueError(f"Could not resolve {expected} distinct allocated CUDA GPU UUIDs")
    return uuids


def probe(expected_gpus: int, samples: int, interval_ms: int) -> dict:
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_executable": sys.executable,
        "expected_gpus": expected_gpus,
        "scope": "capability_probe_not_experiment_evidence",
        "fields": FIELDS,
        "requested_samples": samples,
        "interval_ms": interval_ms,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tools": {t: shutil.which(t) for t in ("nvidia-smi", "dcgmi", "nsys", "ncu")},
        "commands": {},
        "issues": [],
        "assessment": {"status": "blocked_or_inconclusive"},
    }
    commands = report["commands"]
    if not report["slurm_job_id"]:
        report["issues"].append("Run inside a Slurm GPU allocation; no device queries were performed")
        return report
    commands["source_commit"] = run_command(["git", "rev-parse", "HEAD"])
    commands["source_status"] = run_command(["git", "status", "--porcelain"])
    commands["gpu_inventory"] = run_command(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,pci.bus_id", "--format=csv,noheader,nounits"]
    )
    inventory_result = commands["gpu_inventory"]
    if inventory_result["returncode"] != 0:
        report["issues"].append("nvidia-smi inventory unavailable; inspect raw command output")
        return report
    inventory = {}
    for row in csv.reader(io.StringIO(inventory_result["stdout"])):
        if len(row) == 5:
            index, uuid, name, driver, bus = (x.strip() for x in row)
            inventory[uuid.lower()] = {
                "index": index,
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "pci_bus_id": bus,
            }
    try:
        uuids = allocated_gpu_uuids(expected_gpus)
        devices = [inventory[u.lower()] for u in uuids]
        if not all("H100" in d["name"] for d in devices):
            raise ValueError("Selected allocation contains a non-H100 GPU")
    except Exception as exc:
        report["issues"].append(f"Allocation identity could not be verified: {type(exc).__name__}: {exc}")
        return report
    report["allocated_gpus"] = devices
    commands["topology"] = run_command(["nvidia-smi", "topo", "-m"])
    for device in devices:
        commands[f"nvlink_status_{device['index']}"] = run_command(["nvidia-smi", "nvlink", "-s", "-i", device["uuid"]])
    commands["dcgm_version"] = run_command(["dcgmi", "--version"])
    commands["dcgm_discovery"] = run_command(["dcgmi", "discovery", "-l"])
    discovery = commands["dcgm_discovery"]
    if discovery["returncode"] != 0:
        report["issues"].append(
            "DCGM discovery failed: missing tool, unavailable host engine or permissions; inspect raw output"
        )
        return report
    try:
        mapping = dcgm_uuid_mapping(discovery["stdout"])
        gpu_ids = {mapping[u.lower()] for u in uuids}
        if len(gpu_ids) != expected_gpus:
            raise ValueError("Allocated GPUs do not map to distinct DCGM IDs")
    except (KeyError, ValueError) as exc:
        report["issues"].append(f"Cannot safely map allocated UUIDs to DCGM IDs: {exc}")
        return report
    report["dcgm_gpu_ids"] = sorted(gpu_ids)
    for gpu in sorted(gpu_ids):
        commands[f"profile_groups_{gpu}"] = run_command(["dcgmi", "profile", "-l", "-i", str(gpu)])
    commands["dmon"] = run_command(
        [
            "dcgmi",
            "dmon",
            "-e",
            ",".join(FIELDS),
            "-i",
            ",".join(str(g) for g in sorted(gpu_ids)),
            "-d",
            str(interval_ms),
            "-c",
            str(samples),
        ],
        timeout=samples * interval_ms / 1000 + 20,
    )
    result = commands["dmon"]
    report["samples"] = parse_dmon(result["stdout"], gpu_ids)
    report["assessment"] = assess_samples(report["samples"], gpu_ids, command_ok=result["returncode"] == 0)
    if report["assessment"]["status"] != "readable_requires_calibration":
        report["issues"].append(
            "Need >=2 numeric TX and RX samples on every allocated GPU; N/A does not identify the cause"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, choices=(2, 4), default=2)
    parser.add_argument("--samples", type=int, choices=range(3, 31), default=5)
    parser.add_argument("--interval-ms", type=int, choices=(100, 200, 500, 1000), default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = probe(args.expected_gpus, args.samples, args.interval_ms)
    (args.output_dir / "PROBE.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    status = report["assessment"]["status"]
    summary = [
        "# H100 NVLink capability probe",
        "",
        f"Status: **{status}**",
        "",
        "This is an idle capability check, not calibration or benchmark evidence.",
        "Raw command output, identity mappings and samples are in PROBE.json.",
        "",
    ]
    summary.extend(f"- {issue}" for issue in report["issues"])
    summary.extend(
        [
            "",
            "Next: validate with controlled inter-GPU transfers if readable; otherwise inspect the raw errors.",
            "Missing DCGM access does not prove NVLink is unavailable through every API.",
            "",
        ]
    )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary))
    print(f"NVLink probe: {status}\nReport: {args.output_dir / 'PROBE.json'}")
    for issue in report["issues"]:
        print(f"- {issue}")
    return 0 if status == "readable_requires_calibration" else 2


if __name__ == "__main__":
    raise SystemExit(main())
