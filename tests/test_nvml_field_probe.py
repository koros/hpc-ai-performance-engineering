import ctypes
from types import SimpleNamespace

import pytest

from hpc_ai_perf.profiling import nvlink_probe as nvml

UUIDS = ["GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]


def fake_library(monkeypatch, *, field_error=0, query_error=0, stale=False, mismatch=False, init_error=0):
    state = {"queries": 0, "shutdown": 0, "scope_ids": []}

    def get_handle(uuid, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(UUIDS.index(uuid.decode()) + 1)
        return 0

    def get_uuid(handle, buffer, length):
        buffer.value = (UUIDS[0] if mismatch else UUIDS[handle.value - 1]).encode()
        return 0

    def fields(handle, count, output):
        state["queries"] += 1
        for field in output:
            state["scope_ids"].append(field.scope_id)
            field.return_code = field_error
            field.timestamp = 1000 if stale else state["queries"] * 1000
            field.value_type = 3
            field.value.ull = 0
        return query_error

    def shutdown():
        state["shutdown"] += 1
        return 0

    lib = SimpleNamespace(
        nvmlInit_v2=lambda: init_error,
        nvmlShutdown=shutdown,
        nvmlErrorString=lambda code: b"Not supported or permission denied",
        nvmlDeviceGetHandleByUUID=get_handle,
        nvmlDeviceGetUUID=get_uuid,
        nvmlDeviceGetFieldValues=fields,
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda name: lib)
    monkeypatch.setattr(nvml.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    return state


def test_native_field_abi_layout():
    assert ctypes.sizeof(nvml._Field) == 40
    assert nvml._Field.timestamp.offset == 8
    assert nvml._Field.return_code.offset == 28
    assert nvml._Field.value.offset == 32


def test_direct_nvml_uuid_mapping_and_raw_units(monkeypatch):
    state = fake_library(monkeypatch)
    report = nvml.probe_nvml(UUIDS, 3, 1000)
    assert report["assessment"]["status"] == "readable_requires_calibration"
    assert state["shutdown"] == 1
    assert set(state["scope_ids"]) == {0xFFFFFFFF}
    assert report["field_ids"] == [138, 139]
    assert report["unit"].startswith("KiB")
    assert len(report["samples"]) == 6
    assert not report["assessment"]["bandwidth_calibrated"]


@pytest.mark.parametrize("kwargs", [{"field_error": 3}, {"query_error": 4}, {"stale": True}])
def test_errors_or_repeated_cached_timestamps_cannot_pass(monkeypatch, kwargs):
    state = fake_library(monkeypatch, **kwargs)
    report = nvml.probe_nvml(UUIDS, 3, 1000)
    assert report["assessment"]["status"] == "blocked_or_inconclusive"
    assert state["shutdown"] == 1
    assert report["issues"]
    if kwargs.get("field_error"):
        assert report["samples"][0]["fields"][0]["value"] is None
        assert report["samples"][0]["fields"][0]["return_code"] == 3


def test_handle_uuid_mismatch_stops_before_any_read(monkeypatch):
    state = fake_library(monkeypatch, mismatch=True)
    report = nvml.probe_nvml(UUIDS, 3, 1000)
    assert report["assessment"]["status"] == "blocked_or_inconclusive"
    assert state["queries"] == 0
    assert state["shutdown"] == 1


def test_initialization_failure_does_not_shutdown_uninitialized_nvml(monkeypatch):
    state = fake_library(monkeypatch, init_error=4)
    report = nvml.probe_nvml(UUIDS, 3, 1000)
    assert report["assessment"]["status"] == "blocked_or_inconclusive"
    assert state["shutdown"] == state["queries"] == 0


def test_library_missing_is_diagnostic(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")

    def missing(name):
        raise OSError("NVML library missing")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    report = nvml.probe_nvml(UUIDS, 3, 1000)
    assert "library missing" in report["issues"][0]


def test_no_native_library_calls_outside_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: pytest.fail("must not load native library"))
    assert nvml.probe_nvml(UUIDS, 3, 1000)["issues"] == ["Slurm allocation required"]


@pytest.mark.parametrize(
    "value_type,member,value,error",
    [
        (3, "ull", 18_000_000_000_000, None),
        (3, "ull", 2**64 - 1, "NVML unavailable sentinel"),
        (1, "uint", 2**32 - 1, "NVML unavailable sentinel"),
        (4, "sll", -1, "NVML unavailable sentinel"),
        (4, "sll", -2, "Negative driver value"),
        (0, "double", float("nan"), "Non-finite driver value"),
        (0, "double", float("inf"), "Non-finite driver value"),
    ],
)
def test_decoder_retains_driver_values_and_separates_failure_reasons(value_type, member, value, error):
    import json

    field = nvml._Field()
    field.field_id, field.scope_id, field.timestamp = 138, 0xFFFFFFFF, 12345
    field.value_type = value_type
    setattr(field.value, member, value)
    decoded = nvml.decode_field(field, lambda code: "error")
    assert decoded["error"] == error
    assert decoded["raw_value"] == (value if value == value and value != float("inf") else str(value))
    assert decoded["raw_union_hex"] == bytes(field.value).hex()
    assert decoded["value"] == (value if error is None else None)
    json.dumps(decoded, allow_nan=False)


def test_decoder_retains_successful_value_when_timestamp_is_missing():
    field = nvml._Field()
    field.value_type = 3
    field.value.ull = 7
    decoded = nvml.decode_field(field, lambda code: "error")
    assert decoded["raw_value"] == 7
    assert decoded["error"] == "Missing driver timestamp"


def test_failed_api_field_never_interprets_undefined_union():
    field = nvml._Field()
    field.return_code = 4
    field.value.ull = 123
    decoded = nvml.decode_field(field, lambda code: "permission denied")
    assert decoded["value"] is decoded["raw_value"] is decoded["raw_union_hex"] is None
    assert decoded["error"] == "permission denied"
