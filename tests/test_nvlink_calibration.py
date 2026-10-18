import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

UUIDS = ("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture
def calibration():
    spec = importlib.util.spec_from_file_location("calibration", "scripts/calibrate_h100_nvlink.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reading(values, timestamp):
    return {
        "samples": [
            {
                "gpu_uuid": uuid,
                "fields": [
                    {"field_id": field_id, "timestamp_us": timestamp, "value": values.get((uuid, field_id), 0)}
                    for field_id in (138, 139)
                ],
            }
            for uuid in UUIDS
        ]
    }


def test_known_transfer_compares_each_endpoint_without_double_counting(calibration):
    phase = {
        "source_uuid": UUIDS[0],
        "destination_uuid": UUIDS[1],
        "payload_bytes": 64 * 1024 * 1024,
        "copy_correct": True,
        "before": reading({}, 1_000_000),
        "after": reading({(UUIDS[0], 138): 65536, (UUIDS[1], 139): 65536}, 2_000_000),
    }
    result = calibration.analyze_phase(phase, UUIDS)
    assert result["responsive_expected_directions"]
    assert result["source_tx_raw_units_per_payload_byte"] == 1 / 1024
    assert result["destination_rx_raw_units_per_payload_byte"] == 1 / 1024
    phase["copy_correct"] = False
    assert not calibration.analyze_phase(phase, UUIDS)["responsive_expected_directions"]


@pytest.mark.parametrize("case", ["frozen", "stale", "reset", "missing", "wrong_direction"])
def test_bad_counter_evidence_does_not_pass(calibration, case):
    before = reading({(UUIDS[0], 138): 10}, 1_000_000)
    after = reading({(UUIDS[0], 138): 20, (UUIDS[1], 139): 20}, 2_000_000)
    if case == "frozen":
        after = reading({(UUIDS[0], 138): 10}, 2_000_000)
    elif case == "stale":
        after = reading({(UUIDS[0], 138): 20, (UUIDS[1], 139): 20}, 1_000_000)
    elif case == "reset":
        after = reading({(UUIDS[0], 138): 9, (UUIDS[1], 139): 20}, 2_000_000)
    elif case == "missing":
        after["samples"][0]["fields"][0]["value"] = None
    else:
        after = reading({(UUIDS[0], 138): 10, (UUIDS[0], 139): 20, (UUIDS[1], 138): 20}, 2_000_000)
    result = calibration.analyze_phase(
        {
            "source_uuid": UUIDS[0],
            "destination_uuid": UUIDS[1],
            "payload_bytes": 1024,
            "copy_correct": True,
            "before": before,
            "after": after,
        },
        UUIDS,
    )
    assert not result["responsive_expected_directions"]


def test_snapshot_queries_only_explicit_uuids_and_has_timeout(calibration, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(reading({}, 1)), stderr="")

    monkeypatch.setattr(calibration.subprocess, "run", run)
    calibration.snapshot(UUIDS)
    assert all(uuid in calls[0][0] for uuid in UUIDS)
    assert calls[0][1]["timeout"] == 30


def test_failure_still_writes_diagnostic_and_never_claims_bandwidth(calibration, monkeypatch, tmp_path):
    def fail(*args):
        raise RuntimeError("peer access unavailable")

    monkeypatch.setattr(calibration, "run", fail)
    output = tmp_path / "job-123"
    monkeypatch.setattr(sys, "argv", ["calibrate", "--output-dir", str(output)])
    assert calibration.main() == 2
    report = json.loads((output / "CALIBRATION.json").read_text())
    assert report["status"] == "inconclusive"
    assert not report["unit_conversion_validated"]
    assert not report["link_capacity_measured"]
    assert "peer access" in report["issues"][0]
    assert (output / "SUMMARY.md").is_file()
    with pytest.raises(FileExistsError):
        calibration.main()
