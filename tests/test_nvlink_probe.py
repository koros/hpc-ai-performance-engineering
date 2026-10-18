import importlib.util
import json
import subprocess
import sys

import pytest

UUIDS = ("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture
def probe_module():
    spec = importlib.util.spec_from_file_location("nvlink_probe", "scripts/probe_h100_nvlink.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_commands(module, monkeypatch, *, dmon=None, discovery_error=False, wrong_mapping=False):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        result = {"command": command, "returncode": 0, "stdout": "", "stderr": "", "status": "finished"}
        if command[0] == "nvidia-smi" and command[1].startswith("--query-gpu"):
            result["stdout"] = "\n".join(
                f"{i}, {uuid}, NVIDIA H100 SXM 80GB, 550.1, 00000000:0{i}:00.0" for i, uuid in enumerate(UUIDS)
            )
        elif command[:3] == ["dcgmi", "discovery", "-l"]:
            # DCGM IDs deliberately differ from the nvidia-smi/CUDA indices.
            result["stdout"] = f"| 7 | Name: H100 |\n| | Device UUID: {UUIDS[0]} |\n"
            if not wrong_mapping:
                result["stdout"] += f"| 9 | Name: H100 |\n| | Device UUID: {UUIDS[1]} |\n"
            if discovery_error:
                result.update(returncode=1, stderr="Insufficient permissions")
        elif command[:2] == ["dcgmi", "dmon"]:
            result["stdout"] = dmon or "GPU 7 0.000 0.000\nGPU 9 0.000 0.000\n" * 3
        return result

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(module, "run_command", run)
    monkeypatch.setattr(module, "allocated_gpu_uuids", lambda expected: UUIDS)
    return calls


def test_idle_numeric_samples_are_readable_but_not_calibration(probe_module, monkeypatch):
    calls = fake_commands(probe_module, monkeypatch)
    result = probe_module.probe(2, 5, 1000)
    assert result["assessment"]["status"] == "readable_requires_calibration"
    assert not result["assessment"]["positive_traffic_observed"]
    assert not result["assessment"]["bandwidth_calibrated"]
    assert result["dcgm_gpu_ids"] == [7, 9]
    dmon = next(c for c in calls if c[:2] == ["dcgmi", "dmon"])
    assert dmon[dmon.index("-i") + 1] == "7,9"
    assert not any("--pause" in c or "--resume" in c or "sudo" in c for c in calls)


@pytest.mark.parametrize("raw", ["N/A", "nan", "inf", "-1", "9223372036854775792", "140737488355328"])
def test_unavailable_or_sentinel_samples_cannot_pass(probe_module, monkeypatch, raw):
    fake_commands(probe_module, monkeypatch, dmon=f"GPU 7 {raw} 0.0\nGPU 9 0.0 {raw}\n" * 3)
    result = probe_module.probe(2, 5, 1000)
    assert result["assessment"]["status"] == "blocked_or_inconclusive"
    json.dumps(result, allow_nan=False)


def test_unknown_gpu_rows_and_malformed_rows_do_not_supply_coverage(probe_module):
    samples = probe_module.parse_dmon("GPU 4 12 13\nGPU 7 1 2\nGPU 9 1\n# GPU 9 1 2\n", {7, 9})
    assert len(samples) == 1
    assert probe_module.assess_samples(samples, {7, 9}, command_ok=True)["status"] == "blocked_or_inconclusive"
    assert probe_module.assess_samples(samples * 3, {7}, command_ok=False)["status"] == "blocked_or_inconclusive"


@pytest.mark.parametrize("kwargs", [{"discovery_error": True}, {"wrong_mapping": True}])
def test_no_sampling_when_dcgm_identity_or_permissions_fail(probe_module, monkeypatch, kwargs):
    calls = fake_commands(probe_module, monkeypatch, **kwargs)
    result = probe_module.probe(2, 5, 1000)
    assert result["assessment"]["status"] == "blocked_or_inconclusive"
    assert result["issues"]
    assert not any(c[:2] == ["dcgmi", "dmon"] for c in calls)


def test_no_gpu_commands_outside_allocation(probe_module, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(probe_module, "run_command", lambda *a, **k: pytest.fail("must not query devices"))
    result = probe_module.probe(2, 5, 1000)
    assert not result["commands"]
    assert result["assessment"]["status"] == "blocked_or_inconclusive"


def test_conflicting_discovery_uuid_is_rejected(probe_module):
    with pytest.raises(ValueError, match="conflicting"):
        probe_module.dcgm_uuid_mapping(f"| 7 | {UUIDS[0]} |\n| 9 | {UUIDS[0]} |")


def test_missing_cli_and_timeouts_keep_diagnostics(probe_module, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("dcgmi missing")

    monkeypatch.setattr(subprocess, "run", missing)
    assert probe_module.run_command(["dcgmi"])["status"] == "missing_executable"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("dcgmi", 1, output=b"partial", stderr=b"error")

    monkeypatch.setattr(subprocess, "run", timeout)
    result = probe_module.run_command(["dcgmi"])
    assert result["status"] == "timeout"
    assert result["stdout"] == "partial"


def test_cli_saves_report_and_refuses_overwrite(probe_module, monkeypatch, tmp_path):
    fake_commands(probe_module, monkeypatch)
    output = tmp_path / "job-123"
    monkeypatch.setattr(sys, "argv", ["probe", "--output-dir", str(output)])
    assert probe_module.main() == 0
    assert json.loads((output / "PROBE.json").read_text())["assessment"]["status"] == "readable_requires_calibration"
    assert (output / "SUMMARY.md").is_file()
    with pytest.raises(FileExistsError):
        probe_module.main()
