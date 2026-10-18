"""Validate raw controlled-transfer reports before converting NVML deltas to bytes."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_PAYLOAD_MIB = (16, 64, 128)


def _last_fields(snapshot: dict) -> dict:
    result = {}
    for sample in snapshot.get("samples", []):
        for field in sample.get("fields", []):
            result[(sample["gpu_uuid"], field["field_id"])] = field
    return result


def _delta(before: dict, after: dict, uuid: str, field_id: int) -> int | float:
    a = _last_fields(before).get((uuid, field_id))
    b = _last_fields(after).get((uuid, field_id))
    for row in (a, b):
        if row is None or row.get("return_code") != 0 or row.get("error") is not None:
            raise ValueError(f"{uuid} field {field_id}: missing or failed reading")
        value = row.get("value")
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{uuid} field {field_id}: invalid counter value")
        if row.get("scope_id") != 0xFFFFFFFF:
            raise ValueError("Calibration must use aggregate-across-links field scope")
        if type(row.get("timestamp_us")) is not int or row["timestamp_us"] <= 0:
            raise ValueError("Missing driver timestamp")
    if b["timestamp_us"] <= a["timestamp_us"]:
        raise ValueError("Stale driver timestamps")
    if b["value"] < a["value"]:
        raise ValueError("Counter reset/decrease/wrap")
    return b["value"] - a["value"]


def validate_report(record: dict, tolerance: float = 0.02) -> dict:
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.05:
        raise ValueError("Calibration tolerance must be between 0 and 0.05")
    if record.get("status") != "responsive_requires_unit_validation":
        raise ValueError("Transfer diagnostic did not complete responsively")
    uuids = record.get("gpu_uuids", [])
    if len(uuids) != 2 or len(set(uuids)) != 2:
        raise ValueError("Exactly two distinct GPU identities required")
    payload_mib = record.get("payload_mib")
    if type(payload_mib) is not int or payload_mib not in REQUIRED_PAYLOAD_MIB:
        raise ValueError("Unexpected calibration payload size")
    for uuid in uuids:
        for field_id in (138, 139):
            if _delta(record["idle"]["before"], record["idle"]["after"], uuid, field_id) != 0:
                raise ValueError("Idle baseline contains traffic")
    phases = record.get("phases", [])
    if len(phases) != 2 or {(p["source_uuid"], p["destination_uuid"]) for p in phases} != {
        tuple(uuids),
        tuple(reversed(uuids)),
    }:
        raise ValueError("Both transfer directions must be tested exactly once")
    ratios = []
    for phase in phases:
        if phase.get("copy_correct") is not True:
            raise ValueError("Copy correctness check failed")
        copies = phase.get("copies")
        payload = phase.get("payload_bytes")
        size = phase.get("bytes_per_copy")
        if (
            type(copies) is not int
            or copies <= 0
            or type(size) is not int
            or size != payload_mib * 1024 * 1024
            or type(payload) is not int
            or payload != copies * size
        ):
            raise ValueError("Known payload accounting is inconsistent")
        source, destination = phase["source_uuid"], phase["destination_uuid"]
        tx = _delta(phase["before"], phase["after"], source, 138)
        rx = _delta(phase["before"], phase["after"], destination, 139)
        if (
            _delta(phase["before"], phase["after"], source, 139) != 0
            or _delta(phase["before"], phase["after"], destination, 138) != 0
        ):
            raise ValueError("Unexpected reverse-direction traffic during one-way copy")
        for label, delta in (("source_tx", tx), ("destination_rx", rx)):
            ratio = delta * 1024 / payload
            if abs(ratio - 1) > tolerance:
                raise ValueError(f"{label}: KiB conversion does not match known payload ({ratio:.6g})")
            ratios.append(
                {
                    "source_uuid": source,
                    "destination_uuid": destination,
                    "endpoint": label,
                    "converted_bytes_over_known_payload": ratio,
                }
            )
    return {
        "payload_mib": payload_mib,
        "gpu_uuids": uuids,
        "ratios": ratios,
        "driver_inventory": record.get("driver_inventory"),
        "torch_version": record.get("torch_version"),
        "cuda_version": record.get("cuda_version"),
        "script_sha256": record.get("script_sha256"),
    }


def verify_calibrations(root: str | Path) -> dict:
    """Check the latest report for each required size; never fall back past a newer failure."""
    selected = {}
    for path in sorted(Path(root).glob("job-*/CALIBRATION.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unreadable calibration report {path}: {exc}") from exc
        size = record.get("payload_mib")
        if type(size) is int and size in REQUIRED_PAYLOAD_MIB:
            previous = selected.get(size)
            key = (str(record.get("created_at", "")), str(path))
            if previous is None or key > previous[0]:
                selected[size] = (key, path, record)
    missing = sorted(set(REQUIRED_PAYLOAD_MIB) - selected.keys())
    if missing:
        raise ValueError(f"Missing controlled-transfer reports for payload MiB: {missing}")
    reports = []
    for size in REQUIRED_PAYLOAD_MIB:
        _, path, record = selected[size]
        try:
            result = validate_report(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        result.update(path=str(path.resolve()), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reports.append(result)
    return {
        "schema_version": "1.0",
        "validated_at": datetime.now(UTC).isoformat(),
        "status": "unit_conversion_validated_for_recorded_setups",
        "field_ids": [138, 139],
        "scope_id": 0xFFFFFFFF,
        "bytes_per_raw_unit": 1024,
        "relative_tolerance": 0.02,
        "reports": reports,
        "per_link_attribution": False,
        "link_capacity_measured": False,
        "collector_overhead_validated": False,
        "limitation": (
            "Known-copy validation on recorded GPU setups; inference collector still needs timing/overhead checks"
        ),
    }
