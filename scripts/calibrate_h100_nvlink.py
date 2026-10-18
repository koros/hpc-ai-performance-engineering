"""Bounded H100 peer-copy diagnostic: idle baseline and known bidirectional payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from hpc_ai_perf.profiling.system import visible_cuda_device_uuids


def snapshot(uuids: tuple[str, ...]) -> dict:
    command = [sys.executable, "-m", "hpc_ai_perf.profiling.nvlink_probe", "--samples", "3"]
    for uuid in uuids:
        command.extend(["--gpu-uuid", uuid])
    # Retain three samples, including driver timestamps, without resetting counters.
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f"NVML snapshot failed: {result.returncode}: {result.stderr}")
    return json.loads(result.stdout)


def last_fields(reading: dict) -> dict:
    fields = {}
    for sample in reading.get("samples", []):
        for field in sample["fields"]:
            fields[(sample["gpu_uuid"], field["field_id"])] = field
    return fields


def counter_deltas(before: dict, after: dict, uuids: tuple[str, ...]) -> dict:
    first, last = last_fields(before), last_fields(after)
    result = {}
    for uuid in uuids:
        result[uuid] = {}
        for field_id in (138, 139):
            a, b = first.get((uuid, field_id)), last.get((uuid, field_id))
            row = {"raw_delta": None, "elapsed_seconds": None, "error": None}
            if a is None or b is None or a.get("value") is None or b.get("value") is None:
                row["error"] = "Missing or rejected NVML value"
            elif b["timestamp_us"] <= a["timestamp_us"]:
                row["error"] = "Driver timestamp did not advance"
            elif b["value"] < a["value"]:
                row["error"] = "Counter decreased/reset/wrapped; delta not usable"
            else:
                row["raw_delta"] = b["value"] - a["value"]
                row["elapsed_seconds"] = (b["timestamp_us"] - a["timestamp_us"]) / 1e6
            result[uuid][str(field_id)] = row
    return result


def analyze_phase(phase: dict, uuids: tuple[str, ...]) -> dict:
    deltas = counter_deltas(phase["before"], phase["after"], uuids)
    source, destination = phase["source_uuid"], phase["destination_uuid"]
    tx, rx = deltas[source]["138"], deltas[destination]["139"]
    complete = all(not value["error"] for fields in deltas.values() for value in fields.values())
    responsive = complete and phase["copy_correct"] and tx["raw_delta"] > 0 and rx["raw_delta"] > 0
    return {
        "counter_deltas": deltas,
        "responsive_expected_directions": responsive,
        "source_tx_raw_units_per_payload_byte": tx["raw_delta"] / phase["payload_bytes"] if complete else None,
        "destination_rx_raw_units_per_payload_byte": rx["raw_delta"] / phase["payload_bytes"] if complete else None,
    }


def copy_payload(torch, source, destination, duration: float) -> dict:
    torch.cuda.synchronize(source.device)
    torch.cuda.synchronize(destination.device)
    started = time.perf_counter()
    copies = 0
    while time.perf_counter() - started < duration:
        for _ in range(8):
            destination.copy_(source, non_blocking=True)
            copies += 1
        torch.cuda.synchronize(source.device)
        torch.cuda.synchronize(destination.device)
    elapsed = time.perf_counter() - started
    return {
        "copies": copies,
        "payload_bytes": copies * source.numel() * source.element_size(),
        "copy_loop_seconds": elapsed,
        "bytes_per_copy": source.numel() * source.element_size(),
    }


def run(report: dict, duration: float, payload_mib: int) -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run inside a dedicated two-GPU Slurm allocation")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two CUDA-visible GPUs required")
    if not all("H100" in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("H100 allocation required")
    uuids = visible_cuda_device_uuids(2)
    if len(uuids) != 2 or len(set(uuids)) != 2:
        raise RuntimeError("Cannot identify two distinct allocated GPU UUIDs")
    report["gpu_uuids"] = uuids
    report["torch_version"] = str(torch.__version__)
    report["cuda_version"] = torch.version.cuda
    report["peer_access"] = [[torch.cuda.can_device_access_peer(i, j) for j in range(2) if j != i] for i in range(2)]
    if not all(row[0] for row in report["peer_access"]):
        raise RuntimeError("Bidirectional CUDA peer access unavailable")
    buffers = [torch.full((payload_mib * 1024 * 1024,), 17, dtype=torch.uint8, device=f"cuda:{i}") for i in range(2)]
    # Warm both transfer directions before the baseline.
    buffers[1].copy_(buffers[0])
    buffers[0].copy_(buffers[1])
    for i in range(2):
        torch.cuda.synchronize(i)
    time.sleep(1)
    idle = {"before": snapshot(uuids)}
    time.sleep(duration)
    idle["after"] = snapshot(uuids)
    idle["counter_deltas"] = counter_deltas(idle["before"], idle["after"], uuids)
    report["idle"] = idle
    for source_id, destination_id, pattern in ((0, 1, 37), (1, 0, 91)):
        source, destination = buffers[source_id], buffers[destination_id]
        source.fill_(pattern)
        torch.cuda.synchronize(source.device)
        phase = {"source_uuid": uuids[source_id], "destination_uuid": uuids[destination_id], "before": snapshot(uuids)}
        report["phases"].append(phase)
        phase.update(copy_payload(torch, source, destination, duration))
        # Allow delayed field updates to settle; each snapshot also samples for two seconds.
        time.sleep(1)
        phase["after"] = snapshot(uuids)
        phase["copy_correct"] = bool(torch.all(destination == pattern).item())
        phase["analysis"] = analyze_phase(phase, uuids)
    idle_clean = all(
        row["raw_delta"] == 0 and not row["error"]
        for fields in idle["counter_deltas"].values()
        for row in fields.values()
    )
    responsive = all(p["analysis"]["responsive_expected_directions"] for p in report["phases"])
    report["status"] = "responsive_requires_unit_validation" if idle_clean and responsive else "inconclusive"
    if not idle_clean:
        report["issues"].append("Idle baseline changed or was invalid; possible background traffic or delayed updates")
    if not responsive:
        report["issues"].append(
            "Expected source TX and destination RX did not both respond correctly in both directions"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration-seconds", type=int, choices=range(1, 6), default=2)
    parser.add_argument("--payload-mib", type=int, choices=(16, 64, 128), default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "inconclusive",
        "issues": [],
        "phases": [],
        "requested_active_seconds_per_direction": args.duration_seconds,
        "payload_mib": args.payload_mib,
        "scope": "diagnostic_not_canonical_experiment",
        "unit_conversion_validated": False,
        "link_capacity_measured": False,
    }
    try:
        run(report, args.duration_seconds, args.payload_mib)
    except Exception as exc:
        report["issues"].append(f"{type(exc).__name__}: {exc}")
    (args.output_dir / "CALIBRATION.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# H100 NVLink controlled-transfer diagnostic",
        "",
        f"Status: **{report['status']}**",
        "",
        "Raw deltas are not yet converted to bandwidth. TX and RX endpoints must not be added together.",
        "",
    ]
    for phase in report["phases"]:
        if "analysis" in phase:
            lines.append(
                f"- {phase['source_uuid']} -> {phase['destination_uuid']}: "
                f"{phase['payload_bytes']} payload bytes, {phase['analysis']}"
            )
    lines.extend(f"- {issue}" for issue in report["issues"])
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"NVLink transfer diagnostic: {report['status']}\nReport: {args.output_dir / 'CALIBRATION.json'}")
    return 0 if report["status"] == "responsive_requires_unit_validation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
