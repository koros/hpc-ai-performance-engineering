import copy
import hashlib
import json
from pathlib import Path

import pytest

from hpc_ai_perf.validation.nvlink_calibration import validate_report, verify_calibrations

UUIDS = ["GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]


def snapshot(timestamp, increments=None):
    increments = increments or {}
    return {
        "samples": [
            {
                "gpu_uuid": uuid,
                "fields": [
                    {
                        "field_id": fid,
                        "scope_id": 0xFFFFFFFF,
                        "timestamp_us": timestamp,
                        "return_code": 0,
                        "error": None,
                        "value": 13_000_000_000_000 + increments.get((uuid, fid), 0),
                    }
                    for fid in (138, 139)
                ],
            }
            for uuid in UUIDS
        ]
    }


def report(size=64):
    payload = size * 1024 * 1024 * 8
    phases = []
    for index, (source, destination) in enumerate((UUIDS, list(reversed(UUIDS)))):
        before = snapshot((index * 2 + 3) * 1_000_000)
        after = snapshot(
            (index * 2 + 4) * 1_000_000, {(source, 138): payload // 1024, (destination, 139): payload // 1024}
        )
        phases.append(
            {
                "source_uuid": source,
                "destination_uuid": destination,
                "copies": 8,
                "bytes_per_copy": size * 1024 * 1024,
                "payload_bytes": payload,
                "copy_correct": True,
                "before": before,
                "after": after,
            }
        )
    return {
        "status": "responsive_requires_unit_validation",
        "gpu_uuids": UUIDS,
        "payload_mib": size,
        "created_at": "2026-10-18T15:00:00+00:00",
        "idle": {"before": snapshot(1_000_000), "after": snapshot(2_000_000)},
        "phases": phases,
    }


def test_exact_conversion_at_each_endpoint_and_preserves_input():
    data = report()
    original = copy.deepcopy(data)
    validated = validate_report(data)
    assert [r["converted_bytes_over_known_payload"] for r in validated["ratios"]] == [1.0] * 4
    assert validated["driver_inventory"] is None  # Older 64 MiB evidence has no driver metadata.
    assert data == original


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r.update(status="inconclusive"), "did not complete"),
        (lambda r: r["phases"][0].update(payload_bytes=1), "payload accounting"),
        (lambda r: r["phases"][0].update(copy_correct=False), "correctness"),
        (lambda r: r["phases"][0]["after"]["samples"][0]["fields"][0].update(value=13_000_000_000_001), "conversion"),
        (lambda r: r["idle"]["after"]["samples"][0]["fields"][0].update(value=13_000_000_000_001), "Idle"),
        (lambda r: r["phases"][0]["after"]["samples"][0]["fields"][1].update(value=13_000_000_000_001), "reverse"),
        (lambda r: r["phases"][0]["after"]["samples"][0]["fields"][0].update(timestamp_us=1), "Stale"),
        (lambda r: r["phases"][0]["after"]["samples"][0]["fields"][0].update(value=1), "reset"),
        (lambda r: r["phases"][0]["after"]["samples"][0]["fields"][0].update(return_code=3), "failed reading"),
        (lambda r: r["phases"][0]["after"]["samples"][0]["fields"][0].update(scope_id=0), "scope"),
        (lambda r: r["phases"].pop(), "Both transfer directions"),
    ],
)
def test_bad_evidence_is_rejected(mutation, reason):
    data = report()
    mutation(data)
    with pytest.raises(ValueError, match=reason):
        validate_report(data)


def save(root: Path, job: str, data: dict):
    directory = root / f"job-{job}"
    directory.mkdir()
    path = directory / "CALIBRATION.json"
    path.write_text(json.dumps(data))
    return path


def test_requires_all_payload_sizes_and_links_original_evidence_hashes(tmp_path):
    save(tmp_path, "1", report(64))
    with pytest.raises(ValueError, match="16, 128"):
        verify_calibrations(tmp_path)
    paths = [save(tmp_path, "2", report(16)), save(tmp_path, "3", report(128))]
    result = verify_calibrations(tmp_path)
    assert result["bytes_per_raw_unit"] == 1024
    assert not result["collector_overhead_validated"]
    for path in paths:
        entry = next(e for e in result["reports"] if e["path"] == str(path.resolve()))
        assert entry["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_latest_failed_attempt_is_not_silently_replaced_by_older_success(tmp_path):
    for size in (16, 64, 128):
        save(tmp_path, str(size), report(size))
    newer = report(16)
    newer.update(created_at="2026-10-18T16:00:00+00:00", status="inconclusive")
    save(tmp_path, "new", newer)
    with pytest.raises(ValueError, match="job-new"):
        verify_calibrations(tmp_path)
