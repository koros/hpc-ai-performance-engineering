import csv
import subprocess
from types import SimpleNamespace

import pytest

from hpc_ai_perf.slurm.job import SlurmJob
from hpc_ai_perf.slurm.status import submit_job_scripts, write_job_status_csv


def scripts(tmp_path, serial=True):
    for eid in ("E305", "E306", "E307"):
        job = SlurmJob(job_name=eid, command="python run.py", repetitions=3, serial_wave=serial)
        (tmp_path / f"{eid}.sbatch").write_text(job.render())
    return write_job_status_csv(tmp_path, tmp_path / "status.csv")


def test_serial_submission_chains_entire_arrays_and_records_dependencies(tmp_path, monkeypatch):
    status = scripts(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"Submitted batch job {100 + len(calls)}\n")

    monkeypatch.setattr(subprocess, "run", run)
    submitted = submit_job_scripts(tmp_path, status)
    assert calls == [
        ["sbatch", str(tmp_path / "E305.sbatch")],
        ["sbatch", "--dependency=afterany:101", str(tmp_path / "E306.sbatch")],
        ["sbatch", "--dependency=afterany:102", str(tmp_path / "E307.sbatch")],
    ]
    assert [s["job_id"] for s in submitted] == ["101", "102", "103"]
    with status.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["notes"] for r in rows] == ["", "afterany:101", "afterany:102"]
    assert "#SBATCH --array=1-3%1" in (tmp_path / "E305.sbatch").read_text()


def test_serial_dry_run_does_not_submit_or_modify_status(tmp_path, monkeypatch):
    status = scripts(tmp_path)
    original = status.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("must not call sbatch"))
    result = submit_job_scripts(tmp_path, status, dry_run=True)
    assert "--dependency=afterany:<job-id-for-E305>" in result[1]["command"]
    assert status.read_bytes() == original


def test_serial_submission_failure_preserves_successful_jobs(tmp_path, monkeypatch):
    status = scripts(tmp_path)

    def run(command, **kwargs):
        if command[-1].endswith("E306.sbatch"):
            raise subprocess.CalledProcessError(1, command, stderr="submission limit")
        return SimpleNamespace(stdout="Submitted batch job 101\n")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        submit_job_scripts(tmp_path, status)
    with status.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["job_id"] for r in rows] == ["101", "", ""]


def test_serial_submission_rejects_mixed_batch(tmp_path):
    status = scripts(tmp_path)
    (tmp_path / "E306.sbatch").write_text("#!/bin/bash\npython run.py\n")
    with pytest.raises(ValueError, match="cannot mix"):
        submit_job_scripts(tmp_path, status, dry_run=True)


def test_legacy_submission_and_custom_runner_are_unchanged(tmp_path):
    status = scripts(tmp_path, serial=False)
    result = submit_job_scripts(tmp_path, status, runner=lambda script: "Submitted batch job 101")
    assert all("--dependency" not in r["command"] for r in result)
    assert "#SBATCH --array=1-3\n" in (tmp_path / "E305.sbatch").read_text()
