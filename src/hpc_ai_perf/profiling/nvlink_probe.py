"""Read-only NVML field probe, run in a bounded subprocess by the H100 diagnostic."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path


# ABI from NVIDIA's nvmlFieldValue_t / nvmlValue_t documentation.
# https://docs.nvidia.com/deploy/nvml-api/api/structnvmlFieldValue__t.html
class _Value(ctypes.Union):
    _fields_ = [
        ("double", ctypes.c_double),
        ("uint", ctypes.c_uint),
        ("ulong", ctypes.c_ulong),
        ("ull", ctypes.c_ulonglong),
        ("sll", ctypes.c_longlong),
        ("int", ctypes.c_int),
    ]


class _Field(ctypes.Structure):
    _fields_ = [
        ("field_id", ctypes.c_uint),
        ("scope_id", ctypes.c_uint),
        ("timestamp", ctypes.c_longlong),
        ("latency_us", ctypes.c_longlong),
        ("value_type", ctypes.c_int),
        ("return_code", ctypes.c_int),
        ("value", _Value),
    ]


FIELD_IDS = (138, 139)
AGGREGATE_SCOPE = 0xFFFFFFFF


def decode_field(field: _Field, error_string) -> dict:
    result = {
        "field_id": field.field_id,
        "scope_id": field.scope_id,
        "timestamp_us": field.timestamp,
        "latency_us": field.latency_us,
        "return_code": field.return_code,
        "value_type": field.value_type,
        "value": None,
        "error": None,
    }
    if field.return_code != 0:
        result["error"] = error_string(field.return_code)
        return result
    member = {0: "double", 1: "uint", 2: "ulong", 3: "ull", 4: "sll", 5: "int"}.get(field.value_type)
    if member is None:
        result["error"] = "Unrecognized NVML value type"
        return result
    value = getattr(field.value, member)
    if math.isfinite(value) and 0 <= value < 1e13 and field.timestamp > 0:
        result["value"] = value
    else:
        result["error"] = "Invalid value, sentinel or missing driver timestamp"
    return result


def assess(samples: list[dict], uuids: list[str]) -> dict:
    coverage = {}
    for uuid in uuids:
        coverage[uuid] = {}
        for field_id in FIELD_IDS:
            timestamps = {
                f["timestamp_us"]
                for s in samples
                if s["gpu_uuid"] == uuid
                for f in s["fields"]
                if f["field_id"] == field_id and f["value"] is not None
            }
            coverage[uuid][str(field_id)] = len(timestamps)
    ready = bool(uuids) and all(n >= 2 for fields in coverage.values() for n in fields.values())
    return {
        "status": "readable_requires_calibration" if ready else "blocked_or_inconclusive",
        "distinct_valid_samples_by_gpu": coverage,
        "bandwidth_calibrated": False,
        "traffic_attribution_validated": False,
    }


def probe_nvml(uuids: list[str], samples: int, interval_ms: int) -> dict:
    report = {
        "backend": "nvml_field_values",
        "library": "libnvidia-ml.so.1",
        "helper_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "field_ids": list(FIELD_IDS),
        "scope_id": AGGREGATE_SCOPE,
        "unit": "KiB (NVML field definition); raw values, no rate or byte-total conversion",
        "field_semantics": "NVLink data throughput, summed across links; excludes raw/protocol fields",
        "requested_samples": samples,
        "interval_ms": interval_ms,
        "samples": [],
        "issues": [],
        "assessment": {"status": "blocked_or_inconclusive"},
    }
    if not os.environ.get("SLURM_JOB_ID"):
        report["issues"].append("Slurm allocation required")
        return report
    if not uuids or len(set(uuids)) != len(uuids) or any(not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", u) for u in uuids):
        report["issues"].append("Distinct physical GPU UUIDs required")
        return report
    initialized = False
    try:
        lib = ctypes.CDLL(report["library"])
        lib.nvmlErrorString.argtypes = [ctypes.c_int]
        lib.nvmlErrorString.restype = ctypes.c_char_p

        def error_string(code):
            return (lib.nvmlErrorString(code) or b"Unknown NVML error").decode(errors="replace")

        def check(code, operation):
            if code:
                raise RuntimeError(f"{operation}: NVML {code}: {error_string(code)}")

        lib.nvmlInit_v2.argtypes = []
        lib.nvmlInit_v2.restype = ctypes.c_int
        lib.nvmlShutdown.argtypes = []
        lib.nvmlShutdown.restype = ctypes.c_int
        check(lib.nvmlInit_v2(), "initialize")
        initialized = True
        lib.nvmlDeviceGetHandleByUUID.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib.nvmlDeviceGetUUID.restype = ctypes.c_int
        lib.nvmlDeviceGetFieldValues.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_Field)]
        lib.nvmlDeviceGetFieldValues.restype = ctypes.c_int
        handles = {}
        for uuid in uuids:
            handle = ctypes.c_void_p()
            check(lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)), "get UUID handle")
            actual_uuid = ctypes.create_string_buffer(96)
            check(lib.nvmlDeviceGetUUID(handle, actual_uuid, len(actual_uuid)), "verify handle UUID")
            if actual_uuid.value.decode().lower() != uuid.lower():
                raise RuntimeError("NVML handle UUID does not match allocated GPU")
            handles[uuid] = handle
        for index in range(samples):
            if index:
                time.sleep(interval_ms / 1000)
            for uuid, handle in handles.items():
                fields = (_Field * 2)()
                for field, field_id in zip(fields, FIELD_IDS, strict=True):
                    field.field_id, field.scope_id = field_id, AGGREGATE_SCOPE
                code = lib.nvmlDeviceGetFieldValues(handle, len(fields), fields)
                if code:
                    # Per-field output is undefined on a failed overall call.
                    report["issues"].append(f"{uuid}: field query NVML {code}: {error_string(code)}")
                    continue
                report["samples"].append(
                    {
                        "gpu_uuid": uuid,
                        "query_time_ns": time.time_ns(),
                        "fields": [decode_field(f, error_string) for f in fields],
                    }
                )
        report["assessment"] = assess(report["samples"], uuids)
        if report["assessment"]["status"] != "readable_requires_calibration":
            report["issues"].append("Need two distinct driver-timestamped successful TX/RX readings per GPU")
    except (OSError, AttributeError, RuntimeError) as exc:
        report["issues"].append(f"{type(exc).__name__}: {exc}")
    finally:
        if initialized:
            code = lib.nvmlShutdown()
            if code:
                report["issues"].append(f"NVML shutdown returned {code}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", action="append", required=True)
    parser.add_argument("--samples", type=int, choices=range(3, 31), default=5)
    parser.add_argument("--interval-ms", type=int, choices=(100, 200, 500, 1000), default=1000)
    args = parser.parse_args()
    print(json.dumps(probe_nvml(args.gpu_uuid, args.samples, args.interval_ms), allow_nan=False))


if __name__ == "__main__":
    main()
