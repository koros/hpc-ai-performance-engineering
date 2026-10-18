from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from shlex import quote
from typing import Any

from hpc_ai_perf.analysis import analyze_results
from hpc_ai_perf.config import (
    discover_experiment_configs,
    entries_to_json,
    entries_to_table,
    filter_config_entries,
    load_experiment_config,
    load_platform_conditions,
    load_platform_profile,
    materialize_platform_configs,
)
from hpc_ai_perf.planning import plan_serving_layout, plan_sweep
from hpc_ai_perf.readiness import write_readiness_report
from hpc_ai_perf.results import aggregate_results, audit_telemetry, import_result_evidence, verify_result_evidence
from hpc_ai_perf.runners import LocalBenchmarkRunner
from hpc_ai_perf.runtime_checks import format_runtime_checks, run_cluster_runtime_checks, runtime_checks_to_json
from hpc_ai_perf.slurm import (
    BundleResultStatus,
    ExperimentResultStatus,
    append_wave_attempt,
    capture_environment_contract,
    current_wave_attempt,
    format_bundle_result_status,
    load_wave_state,
    manifest_sha256,
    mark_wave_collected,
    prepare_submission_bundle,
    record_job_submission,
    refresh_job_status_csv,
    remove_submission_bundle,
    render_slurm_script,
    submit_job_scripts,
    summarize_bundle_results,
    verify_submission_bundle,
    wave_state_path,
    write_job_status_csv,
    write_job_status_report,
    write_script_checksums,
    write_slurm_script,
    write_submission_manifest,
    write_submission_plan,
)
from hpc_ai_perf.tracking import build_platform_progress_rows, track_experiment_progress, track_platform_progress
from hpc_ai_perf.validation import capability_issues, format_validation_issues, validate_project

LOGGER = logging.getLogger(__name__)
SLURM_PROFILE_KEYS = {
    "account",
    "cpus_per_task",
    "environment",
    "modules",
    "partition",
    "platform",
    "setup_commands",
    "time",
}
SLURM_PROFILE_PLACEHOLDERS = ("replace-with", "TODO", "changeme", "<")


def main() -> None:
    """Run the command-line entry point for the benchmark project.

    The CLI is intentionally a thin orchestration layer. It parses the user's
    command, delegates real work to the relevant package modules, logs where
    generated artefacts were written, and exits early after each command branch.
    Keeping this function as routing glue makes it easier to test the rendering,
    tracking, validation, and analysis logic independently from argparse.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list-configs":
        # Config listing defaults to real E-number matrix configs. Templates and
        # dev/local configs are only shown when the caller opts into them.
        entries = discover_experiment_configs(args.config_root, matrix_only=not args.include_templates)
        entries = filter_config_entries(
            entries,
            phase=args.phase,
            priority=args.priority,
            mode=args.mode,
            workload=args.workload,
            experiment_id=args.experiment_id,
        )
        if args.format == "json":
            print(entries_to_json(entries), end="")
        else:
            print(entries_to_table(entries), end="")
        return

    if args.command == "plan-sweep":
        base_config = load_experiment_config(args.base_config)
        axes = _parse_sweep_axes(args.axis)
        candidates = plan_sweep(base_config, plan_id=args.plan_id, axes=axes)
        payload = {
            "plan_id": args.plan_id,
            "base_config": str(Path(args.base_config)),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            LOGGER.info("Wrote %d sweep candidates to %s", len(candidates), output_path)
        else:
            print(rendered, end="")
        return

    if args.command == "plan-serving-layout":
        config = load_experiment_config(args.config)
        payload = plan_serving_layout(config, visible_devices=args.visible_devices)
        payload["config"] = str(Path(args.config))
        print(json.dumps(payload, indent=2, sort_keys=True) + "\n", end="")
        return

    if args.command == "run":
        _run_local_config(args.config, args.output_dir, repetition_number=args.repetition)
        return

    if args.command == "materialize-platform-configs":
        written_paths = _materialize_platform_batch(
            platform_profile=args.platform_profile,
            execution_matrix=args.execution_matrix,
            config_root=args.config_root,
            output_dir=args.output_dir,
            wave=args.wave,
            experiment_ids=set(args.experiment_id) if args.experiment_id else None,
            include_deferred=args.include_deferred,
        )
        LOGGER.info("Wrote %d resolved platform configs", len(written_paths))
        return

    if args.command == "render-platform-slurm":
        config_paths, script_paths = _render_platform_slurm(
            platform_profile=args.platform_profile,
            execution_matrix=args.execution_matrix,
            config_root=args.config_root,
            config_output_dir=args.config_output_dir,
            output_dir=args.output_dir,
            run_output_dir=args.run_output_dir,
            log_dir=args.log_dir,
            wave=args.wave,
            experiment_ids=set(args.experiment_id) if args.experiment_id else None,
            profile_path=args.slurm_profile,
            partition=args.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
        )
        LOGGER.info("Wrote %d resolved configs and %d Slurm scripts", len(config_paths), len(script_paths))
        return

    if args.command == "create-platform-submission-bundle":
        bundle = _create_platform_submission_bundle(args)
        LOGGER.info("Created submission bundle %s", bundle.root)
        LOGGER.info("Review %s, then submit scripts from %s", bundle.root / "SUBMISSION_PLAN.md", bundle.script_dir)
        return

    if args.command == "submit-platform-wave":
        bundle, state_path, submissions = _submit_platform_wave(args)
        LOGGER.info("Submitted %d job script(s) from %s", len(submissions), bundle.root)
        LOGGER.info(
            "Track this wave with: python -m hpc_ai_perf.cli status-platform-wave --platform-profile %s --wave %s",
            args.platform_profile,
            args.wave,
        )
        LOGGER.info("Saved wave state to %s", state_path)
        return

    if args.command == "status-platform-wave":
        _, attempt, status = _current_platform_wave_status(args)
        scheduler_rows = refresh_job_status_csv(attempt["status_path"])
        scheduler_summary = _summarize_scheduler_rows(scheduler_rows)
        print(format_bundle_result_status(status))
        current_ids = {experiment.experiment_id for experiment in status.experiments}
        historical_ids = {str(experiment_id) for experiment_id in attempt.get("experiment_ids", [])} - current_ids
        if historical_ids:
            print("Policy-deferred historical conditions: " + ", ".join(sorted(historical_ids)))
        print(f"Slurm scheduler states: {scheduler_summary}")
        print(f"Slurm submission record: {attempt['status_path']}")
        return

    if args.command == "collect-platform-wave":
        imported_paths = _collect_platform_wave(args)
        LOGGER.info("Collected %d immutable evidence file(s)", len(imported_paths))
        profile = load_platform_profile(args.platform_profile)
        state = load_wave_state(_wave_state_path_for_args(args, profile.platform_id))
        pending = state.get("incomplete_experiment_ids", [])
        if pending:
            LOGGER.warning("Partial collection; wave remains incomplete for: %s", ", ".join(pending))
        return

    if args.command == "retry-platform-wave":
        profile = load_platform_profile(args.platform_profile)
        state_path = _wave_state_path_for_args(args, profile.platform_id)
        retry_ids = _incomplete_planned_wave_experiment_ids(profile.platform_id, str(args.wave), state_path)
        if not retry_ids:
            raise ValueError("No incomplete planned conditions remain in the current platform-wave policy")
        args.experiment_id = retry_ids
        bundle, state_path, submissions = _submit_platform_wave(args)
        LOGGER.info("Submitted recovery bundle %s for %s", bundle.root, ", ".join(retry_ids))
        LOGGER.info("Saved wave state to %s", state_path)
        LOGGER.info("Submitted %d recovery job script(s)", len(submissions))
        return

    if args.command == "stage-platform-wave":
        staged_paths = _stage_platform_wave(args)
        LOGGER.info("Staged %d evidence path(s); review with git diff --cached --stat", len(staged_paths))
        return

    if args.command == "verify-submission-bundle":
        problems = verify_submission_bundle(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            config_path=args.config,
        )
        if problems:
            print("Submission bundle verification failed:")
            for problem in problems:
                print(f"- {problem}")
            raise SystemExit(1)
        print("Submission bundle verification passed.")
        return

    if args.command == "render-slurm":
        _render_slurm_config(
            args.config,
            output_dir=args.output_dir,
            partition=args.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
            profile_path=args.slurm_profile,
            print_script=args.print,
            repetitions=args.repetitions,
        )
        return

    if args.command == "render-slurm-bulk":
        _render_slurm_bulk(
            config_root=args.config_root,
            output_dir=args.output_dir,
            phase=args.phase,
            priority=args.priority,
            mode=args.mode,
            workload=args.workload,
            experiment_id=args.experiment_id,
            partition=args.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
            profile_path=args.slurm_profile,
            repetitions=args.repetitions,
        )
        return

    if args.command == "plan-slurm-submission":
        plan_path = write_submission_plan(
            args.script_dir,
            args.output,
            batch_name=args.batch_name,
        )
        LOGGER.info("Wrote Slurm submission plan to %s", plan_path)
        return

    if args.command == "init-slurm-status":
        status_path = write_job_status_csv(
            args.script_dir,
            args.output,
            batch_name=args.batch_name,
        )
        LOGGER.info("Wrote Slurm job status CSV to %s", status_path)
        return

    if args.command == "record-slurm-job":
        status_path = record_job_submission(
            args.status,
            experiment_id=args.experiment_id,
            script_path=args.script_path,
            job_id=args.job_id,
            sbatch_output=args.sbatch_output,
            state=args.state,
            submitted_at=args.submitted_at,
            notes=args.notes,
        )
        LOGGER.info("Updated Slurm job status CSV at %s", status_path)
        return

    if args.command == "report-slurm-status":
        report_path = write_job_status_report(args.status, args.output)
        LOGGER.info("Wrote Slurm job status report to %s", report_path)
        return

    if args.command == "submit-slurm-batch":
        # Batch submission can either print sbatch commands for review or submit
        # them and immediately record returned job IDs in the status CSV.
        submissions = submit_job_scripts(
            args.script_dir,
            args.status,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        for submission in submissions:
            if args.dry_run:
                print(submission["command"])
            else:
                print(
                    "{experiment_id}: {job_id} ({script_path})".format(
                        experiment_id=submission["experiment_id"],
                        job_id=submission["job_id"],
                        script_path=submission["script_path"],
                    )
                )
        return

    if args.command == "check-slurm-profile":
        issues = _check_slurm_profile(args.slurm_profile)
        if issues:
            print("Slurm profile is not ready:\n", end="")
            for issue in issues:
                print(f"- {issue}")
            raise SystemExit(1)
        print(f"Slurm profile is ready: {args.slurm_profile}")
        return

    if args.command == "check-runtime":
        checks = run_cluster_runtime_checks(
            expected_gpu_model=args.expected_gpu_model,
            expected_gpus=args.expected_gpus,
            expected_torch_version=args.expected_torch_version,
            expected_torch_cuda=args.expected_torch_cuda,
            require_deepspeed=args.require_deepspeed,
            require_vllm=args.require_vllm,
            require_clean_git=not args.allow_dirty_git,
        )
        if args.format == "json":
            print(runtime_checks_to_json(checks), end="")
        else:
            print(format_runtime_checks(checks), end="")
        if any(check.failed for check in checks):
            raise SystemExit(1)
        return

    if args.command == "aggregate-results":
        result_path = aggregate_results(args.input_dir, args.output)
        LOGGER.info("Wrote aggregated results to %s", result_path)
        return

    if args.command == "audit-telemetry":
        csv_path, markdown_path = audit_telemetry(args.result_root, args.csv_output, args.markdown_output)
        LOGGER.info("Wrote telemetry audit CSV to %s", csv_path)
        LOGGER.info("Wrote telemetry audit Markdown to %s", markdown_path)
        return

    if args.command == "import-result-evidence":
        summary = import_result_evidence(args.source, args.result_root, max_file_bytes=args.max_file_bytes)
        LOGGER.info(
            "Imported %d evidence file(s); %d identical file(s) already present",
            len(summary.imported),
            len(summary.unchanged),
        )
        for path in summary.imported:
            print(path)
        return

    if args.command == "verify-result-evidence":
        problems = verify_result_evidence(args.result_root)
        if problems:
            print("Evidence verification failed:")
            for problem in problems:
                print(f"- {problem}")
            raise SystemExit(1)
        print("Evidence verification passed.")
        return

    if args.command == "analyze-results":
        written_paths = analyze_results(args.input, args.output_dir)
        LOGGER.info("Wrote %d analysis tables to %s", len(written_paths), args.output_dir)
        return

    if args.command == "track-progress":
        csv_path, markdown_path = track_experiment_progress(
            args.config_root,
            args.result_dir,
            args.output_csv,
            args.output_markdown,
            profile=args.profile,
        )
        LOGGER.info("Wrote tracker CSV to %s", csv_path)
        LOGGER.info("Wrote tracker Markdown to %s", markdown_path)
        return

    if args.command == "track-platform-progress":
        csv_path, markdown_path = track_platform_progress(
            args.platform_profile,
            args.execution_matrix,
            args.result_dir,
            args.output_csv,
            args.output_markdown,
            config_root=args.config_root,
            wave=args.wave,
        )
        LOGGER.info("Wrote platform tracker CSV to %s", csv_path)
        LOGGER.info("Wrote platform tracker Markdown to %s", markdown_path)
        return

    if args.command == "validate":
        # Validation prints a human-readable report by default, but can emit JSON
        # for automation in CI or cluster-side health checks.
        issues = validate_project(
            args.config_root,
            args.result_dir,
            include_results=not args.skip_results,
            research_ready=args.research_ready,
        )
        if args.format == "json":
            print(json.dumps([issue.to_dict() for issue in issues], indent=2) + "\n", end="")
        else:
            print(format_validation_issues(issues), end="")
        if issues:
            raise SystemExit(1)
        return

    if args.command == "refresh":
        _refresh_outputs(
            config_root=args.config_root,
            result_dir=args.result_dir,
            processed_results=args.processed_results,
            analysis_dir=args.analysis_dir,
            tracker_csv=args.tracker_csv,
            tracker_markdown=args.tracker_markdown,
            readiness_report=args.readiness_report,
            skip_validation=args.skip_validation,
        )
        return

    if args.command == "readiness-report":
        report_path = write_readiness_report(
            args.output,
            config_root=args.config_root,
            result_dir=args.result_dir,
            slurm_smoke_dir=args.slurm_smoke_dir,
        )
        LOGGER.info("Wrote readiness report to %s", report_path)
        return

    parser.error("A command is required.")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the full argparse command tree used by the project CLI.

    Each subcommand mirrors one operational workflow: local execution, Slurm
    script rendering, submission tracking, result aggregation, analysis,
    progress tracking, validation, or readiness reporting. Defaults point at the
    repository's conventional `configs/`, `results/raw/`, and
    `results/processed/` paths so common project commands stay short.
    """
    parser = argparse.ArgumentParser(description="Manage and run HPC AI benchmark experiments.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment config locally.")
    run_parser.add_argument("--config", required=True, help="Path to a JSON or YAML experiment config.")
    run_parser.add_argument("--output-dir", default=None, help="Override the configured output directory.")
    run_parser.add_argument(
        "--repetition",
        type=int,
        default=None,
        help="One-based independent-trial number; Slurm arrays set this automatically.",
    )

    list_parser = subparsers.add_parser("list-configs", help="List available experiment configs.")
    list_parser.add_argument("--config-root", default="configs", help="Root directory containing config files.")
    list_parser.add_argument("--include-templates", action="store_true", help="Include non-matrix template configs.")
    list_parser.add_argument("--phase", default=None, help="Filter by metadata phase.")
    list_parser.add_argument("--priority", default=None, help="Filter by metadata priority.")
    list_parser.add_argument("--mode", choices=["training", "inference", "profiling"], default=None)
    list_parser.add_argument("--workload", default=None, help="Filter by workload name.")
    list_parser.add_argument("--experiment-id", default=None, help="Filter by experiment ID, such as E001.")
    list_parser.add_argument("--format", choices=["table", "json"], default="table")

    sweep_parser = subparsers.add_parser(
        "plan-sweep",
        help="Dry-run a deterministic, capability-filtered parameter sweep without writing experiment configs.",
    )
    sweep_parser.add_argument("--base-config", required=True, help="Existing config to copy and vary.")
    sweep_parser.add_argument("--plan-id", required=True, help="Stable name for this planned sweep.")
    sweep_parser.add_argument(
        "--axis",
        action="append",
        required=True,
        metavar="PATH=JSON_ARRAY",
        help="Dotted config path and JSON array, for example workload.parameters.prompt_length=[128,512].",
    )
    sweep_parser.add_argument("--output", default=None, help="Optional JSON output path; stdout is used otherwise.")

    serving_plan_parser = subparsers.add_parser(
        "plan-serving-layout",
        help="Dry-run serving GPU groups and routing without importing or launching vLLM.",
    )
    serving_plan_parser.add_argument("--config", required=True, help="Serving experiment config to inspect.")
    serving_plan_parser.add_argument(
        "--visible-devices",
        default=None,
        help="Optional scheduler-visible GPU identifiers, such as 0,1,2,3.",
    )

    materialize_parser = subparsers.add_parser(
        "materialize-platform-configs",
        help="Resolve matrix experiments for an L40S or H100 execution profile.",
    )
    _add_platform_options(materialize_parser)
    materialize_parser.add_argument(
        "--output-dir",
        default=None,
        help="Resolved config directory; defaults to build/platform_configs/<platform>.",
    )
    materialize_parser.add_argument(
        "--include-deferred",
        action="store_true",
        help="Also materialize explicitly deferred conditions for inspection only.",
    )

    platform_slurm_parser = subparsers.add_parser(
        "render-platform-slurm",
        help="Materialize one platform batch and render its planned Slurm jobs.",
    )
    _add_platform_options(platform_slurm_parser)
    platform_slurm_parser.add_argument(
        "--config-output-dir",
        default=None,
        help="Resolved config directory; defaults to build/platform_configs/<platform>.",
    )
    platform_slurm_parser.add_argument(
        "--output-dir",
        default=None,
        help="Slurm script directory; defaults to slurm/generated/<platform>/wave_<wave>.",
    )
    platform_slurm_parser.add_argument(
        "--run-output-dir",
        default=None,
        help=(
            "Directory where generated jobs write result JSON before import; "
            "defaults to build/platform_results/<platform>/wave_<wave>."
        ),
    )
    platform_slurm_parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for Slurm stdout/stderr files; defaults to <script-dir>/logs.",
    )
    platform_slurm_parser.add_argument("--slurm-profile", default=None, help="Override the platform Slurm profile.")
    platform_slurm_parser.add_argument("--partition", default=None, help="Override the platform partition.")
    platform_slurm_parser.add_argument("--account", default=None, help="Optional Slurm account override.")
    platform_slurm_parser.add_argument("--time", default=None, help="Optional Slurm time limit override.")
    platform_slurm_parser.add_argument("--cpus-per-task", type=int, default=None)
    platform_slurm_parser.add_argument("--module", action="append", default=[], help="Additional module to load.")
    platform_slurm_parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional environment variable in KEY=VALUE form.",
    )

    bundle_parser = subparsers.add_parser(
        "create-platform-submission-bundle",
        help="Create an immutable source/configuration bundle for one platform Slurm wave.",
    )
    _add_platform_options(bundle_parser)
    _add_bundle_submission_options(bundle_parser)

    wave_submit_parser = subparsers.add_parser(
        "submit-platform-wave",
        help="Create, record, and submit one immutable platform-wave bundle.",
    )
    _add_platform_options(wave_submit_parser)
    _add_bundle_submission_options(wave_submit_parser)
    wave_submit_parser.add_argument("--state-file", default=None, help="Optional ignored local state-file override.")

    for command, help_text in [
        ("status-platform-wave", "Report validated result coverage for the latest saved platform-wave attempt."),
        ("collect-platform-wave", "Import a complete platform wave without manual bundle copying."),
        ("retry-platform-wave", "Submit a new immutable bundle for incomplete experiments only."),
        ("stage-platform-wave", "Stage exactly the collected raw evidence for one platform wave."),
    ]:
        lifecycle_parser = subparsers.add_parser(command, help=help_text)
        if command == "retry-platform-wave":
            _add_platform_options(lifecycle_parser)
        else:
            lifecycle_parser.add_argument(
                "--platform-profile", required=True, help="Path to platforms/l40s.json or h100.json."
            )
            lifecycle_parser.add_argument("--wave", required=True, help="Wave identifier, such as 0, 1, or 4b.")
        lifecycle_parser.add_argument("--state-file", default=None, help="Optional ignored local state-file override.")
        if command == "collect-platform-wave":
            lifecycle_parser.add_argument(
                "--result-root", default="results/raw", help="Canonical raw evidence directory."
            )
            lifecycle_parser.add_argument(
                "--allow-incomplete",
                action="store_true",
                help="Collect validated completed trials while retaining incomplete wave status.",
            )
        if command == "retry-platform-wave":
            _add_bundle_submission_options(lifecycle_parser)

    verify_bundle_parser = subparsers.add_parser(
        "verify-submission-bundle",
        help="Verify a bundle manifest, snapshot, resolved config, and runtime environment contract.",
    )
    verify_bundle_parser.add_argument("--manifest", required=True, help="Path to SUBMISSION_MANIFEST.json.")
    verify_bundle_parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="Manifest SHA-256 embedded in the generated Slurm script.",
    )
    verify_bundle_parser.add_argument("--config", required=True, help="Resolved config selected by this Slurm script.")

    slurm_parser = subparsers.add_parser("render-slurm", help="Render an sbatch script for an experiment config.")
    slurm_parser.add_argument("--config", required=True, help="Path to a JSON or YAML experiment config.")
    _add_slurm_options(slurm_parser)
    slurm_parser.add_argument("--print", action="store_true", help="Print rendered script instead of writing it.")

    bulk_parser = subparsers.add_parser("render-slurm-bulk", help="Render sbatch scripts for filtered configs.")
    bulk_parser.add_argument("--config-root", default="configs", help="Root directory containing config files.")
    bulk_parser.add_argument("--phase", default=None, help="Filter by metadata phase.")
    bulk_parser.add_argument("--priority", default=None, help="Filter by metadata priority.")
    bulk_parser.add_argument("--mode", choices=["training", "inference", "profiling"], default=None)
    bulk_parser.add_argument("--workload", default=None, help="Filter by workload name.")
    bulk_parser.add_argument("--experiment-id", default=None, help="Filter by experiment ID, such as E001.")
    _add_slurm_options(bulk_parser)

    plan_slurm_parser = subparsers.add_parser(
        "plan-slurm-submission",
        help="Write a reviewable submission plan for generated sbatch scripts.",
    )
    plan_slurm_parser.add_argument("--script-dir", required=True, help="Directory containing generated sbatch scripts.")
    plan_slurm_parser.add_argument("--output", required=True, help="Path for the generated Markdown submission plan.")
    plan_slurm_parser.add_argument("--batch-name", default=None, help="Optional display name for the batch.")

    init_status_parser = subparsers.add_parser(
        "init-slurm-status",
        help="Initialize a CSV for tracking generated sbatch scripts through Slurm submission.",
    )
    init_status_parser.add_argument(
        "--script-dir", required=True, help="Directory containing generated sbatch scripts."
    )
    init_status_parser.add_argument("--output", required=True, help="Path for the generated status CSV.")
    init_status_parser.add_argument("--batch-name", default=None, help="Optional display name for the batch.")

    record_status_parser = subparsers.add_parser(
        "record-slurm-job",
        help="Record a Slurm job ID against an experiment in a status CSV.",
    )
    record_status_parser.add_argument("--status", required=True, help="Path to a Slurm job status CSV.")
    record_status_parser.add_argument("--experiment-id", default=None, help="Experiment ID to update, such as E001.")
    record_status_parser.add_argument("--script-path", default=None, help="Script path to update.")
    record_status_parser.add_argument("--job-id", default=None, help="Slurm job ID.")
    record_status_parser.add_argument(
        "--sbatch-output",
        default=None,
        help="Raw sbatch output, for example 'Submitted batch job 12345'.",
    )
    record_status_parser.add_argument("--state", default="SUBMITTED", help="State to record for the job.")
    record_status_parser.add_argument("--submitted-at", default=None, help="Optional submission timestamp.")
    record_status_parser.add_argument("--notes", default=None, help="Optional notes for the status row.")

    report_status_parser = subparsers.add_parser(
        "report-slurm-status",
        help="Write a Markdown report from a Slurm job status CSV.",
    )
    report_status_parser.add_argument("--status", required=True, help="Path to a Slurm job status CSV.")
    report_status_parser.add_argument("--output", required=True, help="Path for the generated Markdown report.")

    submit_batch_parser = subparsers.add_parser(
        "submit-slurm-batch",
        help="Submit generated sbatch scripts and record returned Slurm job IDs.",
    )
    submit_batch_parser.add_argument(
        "--script-dir", required=True, help="Directory containing generated sbatch scripts."
    )
    submit_batch_parser.add_argument("--status", required=True, help="Path to a Slurm job status CSV.")
    submit_batch_parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting.")
    submit_batch_parser.add_argument("--limit", type=int, default=None, help="Submit only the first N scripts.")

    check_profile_parser = subparsers.add_parser(
        "check-slurm-profile",
        help="Validate that a Slurm profile is ready for real cluster rendering.",
    )
    check_profile_parser.add_argument("--slurm-profile", required=True, help="Path to a Slurm profile JSON file.")

    runtime_parser = subparsers.add_parser(
        "check-runtime",
        help="Check the active GPU node runtime before submitting or accepting HPC evidence.",
    )
    runtime_parser.add_argument(
        "--expected-gpu-model", default=None, help="Substring expected in each CUDA device name."
    )
    runtime_parser.add_argument("--expected-gpus", type=int, default=None, help="Exact visible CUDA device count.")
    runtime_parser.add_argument("--expected-torch-version", default=None, help="Expected torch version prefix.")
    runtime_parser.add_argument("--expected-torch-cuda", default=None, help="Expected torch.version.cuda string.")
    runtime_parser.add_argument(
        "--require-deepspeed", action="store_true", help="Require DeepSpeed import and CUDA accelerator detection."
    )
    runtime_parser.add_argument("--require-vllm", action="store_true", help="Require vLLM import compatibility.")
    runtime_parser.add_argument(
        "--allow-dirty-git", action="store_true", help="Do not fail when the working tree is dirty."
    )
    runtime_parser.add_argument("--format", choices=["table", "json"], default="table")

    aggregate_parser = subparsers.add_parser("aggregate-results", help="Aggregate raw result JSON files into CSV.")
    aggregate_parser.add_argument(
        "--input-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    aggregate_parser.add_argument(
        "--output",
        default="results/processed/experiments.csv",
        help="Path for the processed experiment CSV.",
    )

    telemetry_parser = subparsers.add_parser(
        "audit-telemetry",
        help="Label GPU telemetry provenance and flag repeated-trial outliers without changing raw evidence.",
    )
    telemetry_parser.add_argument("--result-root", default="results/raw")
    telemetry_parser.add_argument("--csv-output", default="results/analysis/telemetry_audit.csv")
    telemetry_parser.add_argument("--markdown-output", default="results/analysis/telemetry_audit.md")

    import_evidence_parser = subparsers.add_parser(
        "import-result-evidence",
        help="Safely import copied HPC matrix JSON files into platform evidence directories.",
    )
    import_evidence_parser.add_argument("--source", required=True, help="Copied result JSON file or directory.")
    import_evidence_parser.add_argument("--result-root", default="results/raw")
    import_evidence_parser.add_argument("--max-file-bytes", type=int, default=5 * 1024 * 1024)

    verify_evidence_parser = subparsers.add_parser(
        "verify-result-evidence",
        help="Validate platform evidence and verify its SHA-256 manifests.",
    )
    verify_evidence_parser.add_argument("--result-root", default="results/raw")

    analyze_parser = subparsers.add_parser("analyze-results", help="Build analysis tables from processed results.")
    analyze_parser.add_argument(
        "--input",
        default="results/processed/experiments.csv",
        help="Processed experiment CSV created by aggregate-results.",
    )
    analyze_parser.add_argument("--output-dir", default="results/analysis", help="Directory for analysis CSV tables.")

    tracker_parser = subparsers.add_parser("track-progress", help="Build tracker reports from configs and raw results.")
    tracker_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    tracker_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    tracker_parser.add_argument(
        "--output-csv",
        default="results/processed/experiment_tracker.csv",
        help="Path for the generated tracker CSV.",
    )
    tracker_parser.add_argument(
        "--output-markdown",
        default="results/processed/experiment_tracker.md",
        help="Path for the generated tracker Markdown.",
    )
    tracker_parser.add_argument(
        "--profile",
        choices=["matrix", "dev", "all"],
        default="matrix",
        help="Config/result profile to include in the generated tracker.",
    )

    platform_tracker_parser = subparsers.add_parser(
        "track-platform-progress",
        help="Track validated trials for one L40S or H100 execution plan.",
    )
    platform_tracker_parser.add_argument("--platform-profile", required=True)
    platform_tracker_parser.add_argument("--execution-matrix", default="platforms/execution_matrix.json")
    platform_tracker_parser.add_argument("--config-root", default="configs")
    platform_tracker_parser.add_argument("--result-dir", default="results/raw")
    platform_tracker_parser.add_argument("--output-csv", default=None)
    platform_tracker_parser.add_argument("--output-markdown", default=None)
    platform_tracker_parser.add_argument("--wave", default=None)

    validate_parser = subparsers.add_parser("validate", help="Validate experiment configs and raw result records.")
    validate_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    validate_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    validate_parser.add_argument("--skip-results", action="store_true", help="Only validate experiment configs.")
    validate_parser.add_argument(
        "--research-ready",
        action="store_true",
        help="Also reject configs whose declared factors do not have a validated execution path.",
    )
    validate_parser.add_argument("--format", choices=["table", "json"], default="table")

    refresh_parser = subparsers.add_parser("refresh", help="Validate and rebuild processed project outputs.")
    refresh_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    refresh_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    refresh_parser.add_argument(
        "--processed-results",
        default="results/processed/experiments.csv",
        help="Path for the aggregated experiment CSV.",
    )
    refresh_parser.add_argument("--analysis-dir", default="results/analysis", help="Directory for analysis CSV tables.")
    refresh_parser.add_argument(
        "--tracker-csv",
        default="results/processed/experiment_tracker.csv",
        help="Path for the generated tracker CSV.",
    )
    refresh_parser.add_argument(
        "--tracker-markdown",
        default="results/processed/experiment_tracker.md",
        help="Path for the generated tracker Markdown.",
    )
    refresh_parser.add_argument(
        "--readiness-report",
        default="results/processed/readiness_report.md",
        help="Path for the generated readiness report.",
    )
    refresh_parser.add_argument("--skip-validation", action="store_true", help="Rebuild outputs without validation.")

    readiness_parser = subparsers.add_parser("readiness-report", help="Write an HPC submission readiness report.")
    readiness_parser.add_argument(
        "--config-root", default="configs", help="Root directory containing experiment configs."
    )
    readiness_parser.add_argument(
        "--result-dir", default="results/raw", help="Directory containing raw result JSON files."
    )
    readiness_parser.add_argument(
        "--slurm-smoke-dir",
        default="slurm/smoke",
        help="Directory containing Slurm smoke preset scripts.",
    )
    readiness_parser.add_argument(
        "--output",
        default="results/processed/readiness_report.md",
        help="Path for the generated readiness report.",
    )
    return parser


def _add_slurm_options(parser: argparse.ArgumentParser) -> None:
    """Attach shared Slurm rendering options to one parser.

    Single-config and bulk rendering both need the same scheduler settings.
    Centralising the options prevents the two commands from drifting apart when
    new site-level fields such as modules, environment variables, or profiles
    are added.
    """
    parser.add_argument("--output-dir", default="slurm/generated", help="Directory for generated sbatch scripts.")
    parser.add_argument("--slurm-profile", default=None, help="Optional JSON file with reusable Slurm defaults.")
    parser.add_argument("--partition", default=None, help="Optional Slurm partition.")
    parser.add_argument("--account", default=None, help="Optional Slurm account.")
    parser.add_argument("--time", default=None, help="Override Slurm time limit, for example 02:00:00.")
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Render each config as a Slurm array with this many independent trials.",
    )
    parser.add_argument("--module", action="append", default=[], help="Module to load; can be repeated.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable KEY=VALUE; can be repeated.",
    )


def _parse_sweep_axes(raw_axes: list[str]) -> dict[str, list[Any]]:
    """Parse repeated PATH=JSON_ARRAY arguments for ``plan-sweep``."""
    axes: dict[str, list[Any]] = {}
    for raw_axis in raw_axes:
        if "=" not in raw_axis:
            raise ValueError(f"Sweep axis must use PATH=JSON_ARRAY syntax: {raw_axis}")
        path, raw_values = raw_axis.split("=", 1)
        path = path.strip()
        if not path:
            raise ValueError(f"Sweep axis path must not be empty: {raw_axis}")
        try:
            values = json.loads(raw_values)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Sweep axis values must be valid JSON: {raw_axis}") from exc
        if not isinstance(values, list):
            raise ValueError(f"Sweep axis values must be a JSON array: {raw_axis}")
        if path in axes:
            raise ValueError(f"Sweep axis is repeated: {path}")
        axes[path] = values
    return axes


def _add_platform_options(parser: argparse.ArgumentParser) -> None:
    """Attach matrix/profile selectors shared by platform execution commands."""
    parser.add_argument("--platform-profile", required=True, help="Path to platforms/l40s.json or h100.json.")
    parser.add_argument(
        "--execution-matrix",
        default="platforms/execution_matrix.json",
        help="Platform wave/defer policy.",
    )
    parser.add_argument("--config-root", default="configs", help="Root containing the logical E-number configs.")
    parser.add_argument("--wave", default=None, help="Wave identifier to select, such as 0, 1, or 4b.")
    parser.add_argument(
        "--experiment-id",
        action="append",
        default=[],
        help="Limit execution to one E-number; may be repeated.",
    )


def _add_bundle_submission_options(parser: argparse.ArgumentParser) -> None:
    """Attach the scheduler settings needed to freeze and submit a bundle."""
    parser.add_argument(
        "--after-job",
        default="",
        type=_numeric_predecessor,
        help="Wait for this Slurm job/array to terminate before starting the wave (afterany).",
    )
    parser.add_argument(
        "--bundle-root",
        default=None,
        help="Parent for immutable bundles; defaults to ../hpc-ai-submissions beside the repository.",
    )
    parser.add_argument(
        "--bundle-id",
        default=None,
        help="Optional stable bundle identifier; defaults to platform-wave-commit-timestamp.",
    )
    parser.add_argument(
        "--venv-activate",
        default=None,
        help="Virtualenv activation script shared by snapshot jobs; defaults to <repository>/.venv/bin/activate.",
    )
    parser.add_argument("--slurm-profile", default=None, help="Override the platform Slurm profile.")
    parser.add_argument("--partition", default=None, help="Override the platform partition.")
    parser.add_argument("--account", default=None, help="Optional Slurm account override.")
    parser.add_argument("--time", default=None, help="Optional Slurm time limit override.")
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--module", action="append", default=[], help="Additional module to load.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional environment variable in KEY=VALUE form.",
    )


def _run_local_config(config_path: str, output_dir: str | None, *, repetition_number: int | None = None) -> None:
    """Run one experiment configuration through the local benchmark runner.

    This path is used for CPU/MPS development, smoke tests, and any simple
    single-process workload execution. It does not submit anything to Slurm; it
    loads the config, executes the registered workload locally, and writes the
    resulting JSON record to either the override directory or the config's
    configured output directory.
    """
    config = load_experiment_config(config_path)
    semantic_issues = capability_issues(config) if config.experiment_id.startswith("E") else []
    if semantic_issues:
        raise RuntimeError("Experiment is not research-ready:\n- " + "\n- ".join(semantic_issues))
    if repetition_number is not None and repetition_number < 1:
        raise ValueError("--repetition must be at least 1")
    runner = LocalBenchmarkRunner()
    result = runner.run(config, repetition_number=repetition_number)
    rank = int(__import__("os").environ.get("RANK", "0"))
    if rank == 0:
        result_path = result.write_json(
            output_dir or config.runtime.output_dir,
            preserve_trial=config.experiment_id.startswith("E"),
        )
        LOGGER.info("Wrote result to %s", result_path)


def _materialize_platform_batch(
    *,
    platform_profile: str,
    execution_matrix: str,
    config_root: str,
    output_dir: str | None,
    wave: str | None,
    experiment_ids: set[str] | None,
    include_deferred: bool,
) -> list[Path]:
    """Resolve selected logical experiments into immutable platform-labelled configs."""
    profile = load_platform_profile(platform_profile)
    conditions = load_platform_conditions(
        execution_matrix,
        profile.platform_id,
        wave=wave,
        experiment_ids=experiment_ids,
        include_deferred=include_deferred,
    )
    if not conditions:
        raise ValueError(f"No {profile.platform_id} conditions matched the requested platform batch")
    resolved_dir = Path(output_dir or f"build/platform_configs/{profile.platform_id}")
    return materialize_platform_configs(
        profile,
        conditions,
        resolved_dir,
        config_root=config_root,
        include_deferred=include_deferred,
    )


def _render_platform_slurm(
    *,
    platform_profile: str,
    execution_matrix: str,
    config_root: str,
    config_output_dir: str | None,
    output_dir: str | None,
    run_output_dir: str | None,
    log_dir: str | None,
    wave: str | None,
    experiment_ids: set[str] | None,
    profile_path: str | None,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
    setup_commands_override: list[str] | None = None,
    environment_overrides: dict[str, str] | None = None,
    submission_manifest: Path | None = None,
    submission_manifest_sha256: str | None = None,
    reuse_resolved_configs: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Resolve and render a platform batch without changing logical matrix configs."""
    profile = load_platform_profile(platform_profile)
    conditions = load_platform_conditions(
        execution_matrix,
        profile.platform_id,
        wave=wave,
        experiment_ids=experiment_ids,
        include_deferred=False,
    )
    conditions = [condition for condition in conditions if condition.status == "planned"]
    if not conditions:
        raise ValueError(f"No planned {profile.platform_id} conditions matched the requested platform batch")

    resolved_dir = Path(config_output_dir or f"build/platform_configs/{profile.platform_id}")
    if reuse_resolved_configs:
        config_paths = [resolved_dir / f"{condition.matrix_id}.json" for condition in conditions]
        missing_configs = [path for path in config_paths if not path.is_file()]
        if missing_configs:
            rendered = ", ".join(str(path) for path in missing_configs)
            raise FileNotFoundError(f"Frozen platform configs are missing: {rendered}")
    else:
        config_paths = materialize_platform_configs(profile, conditions, resolved_dir, config_root=config_root)
    condition_by_id = {condition.matrix_id: condition for condition in conditions}
    slurm_dir = Path(output_dir or f"slurm/generated/{profile.platform_id}/wave_{wave or 'all'}")
    staged_result_dir = Path(run_output_dir or f"build/platform_results/{profile.platform_id}/wave_{wave or 'all'}")
    slurm_log_dir = Path(log_dir or slurm_dir / "logs")
    slurm_log_dir.mkdir(parents=True, exist_ok=True)
    slurm_profile = profile_path or profile.slurm_profile
    slurm_options = _resolve_slurm_options(
        profile_path=slurm_profile,
        partition=partition or profile.partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        env_values=env_values,
    )
    if setup_commands_override is not None:
        slurm_options["setup_commands"] = setup_commands_override
    if environment_overrides:
        slurm_options["environment"] = {**slurm_options["environment"], **environment_overrides}

    script_paths: list[Path] = []
    for config_path in config_paths:
        config = load_experiment_config(config_path)
        semantic_issues = capability_issues(config)
        if semantic_issues:
            raise ValueError(f"{config.experiment_id} is not research-ready: {'; '.join(semantic_issues)}")
        _verify_profile_platform(config, slurm_options.get("platform"), slurm_profile)
        condition = condition_by_id[config.experiment_id]
        allocated_gpus = max(1, config.hardware.gpus, int(config.metadata.get("allocation_gpus", 0)))
        requested_cpus = cpus_per_task or profile.cpus_per_gpu * allocated_gpus
        pre_run_command = None
        if submission_manifest is not None and submission_manifest_sha256 is not None:
            pre_run_command = (
                "python -m hpc_ai_perf.cli verify-submission-bundle "
                f"--manifest {quote(str(submission_manifest))} "
                f"--expected-manifest-sha256 {quote(submission_manifest_sha256)} "
                f"--config {quote(str(config_path))}"
            )
        script = render_slurm_script(
            config,
            config_path,
            partition=slurm_options["partition"],
            account=slurm_options["account"],
            time_limit=slurm_options["time_limit"],
            cpus_per_task=requested_cpus,
            modules=slurm_options["modules"],
            environment={**slurm_options["environment"], **config.runtime.environment},
            setup_commands=slurm_options["setup_commands"],
            repetitions=condition.required_trials,
            run_output_dir=staged_result_dir,
            log_dir=slurm_log_dir,
            pre_run_command=pre_run_command,
        )
        script_paths.append(write_slurm_script(script, slurm_dir, config.experiment_id))
    return config_paths, script_paths


def _create_platform_submission_bundle(args: argparse.Namespace):
    """Freeze one platform wave into a detached source/configuration submission bundle."""
    repository = Path.cwd().resolve()
    default_root = repository.parent / "hpc-ai-submissions"
    bundle_root = Path(args.bundle_root).expanduser() if args.bundle_root else default_root
    provisional_id = args.bundle_id
    if provisional_id is None:
        import datetime

        commit = _git_head(repository)
        timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        profile = load_platform_profile(args.platform_profile)
        provisional_id = f"{profile.platform_id}-wave-{args.wave or 'all'}-{commit[:8]}-{timestamp}".lower()
    bundle = prepare_submission_bundle(repository=repository, bundle_root=bundle_root, bundle_id=provisional_id)
    try:
        source_profile = _bundle_snapshot_path(bundle, args.platform_profile)
        source_matrix = _bundle_snapshot_path(bundle, args.execution_matrix)
        source_config_root = _bundle_snapshot_path(bundle, args.config_root)
        profile = load_platform_profile(source_profile)
        selected_profile = _bundle_snapshot_path(bundle, args.slurm_profile) if args.slurm_profile else None
        effective_profile = selected_profile or (bundle.repo_dir / profile.slurm_profile)
        if not effective_profile.is_file():
            raise FileNotFoundError(f"Slurm profile is missing from submission snapshot: {effective_profile}")

        config_paths = _materialize_platform_batch(
            platform_profile=str(source_profile),
            execution_matrix=str(source_matrix),
            config_root=str(source_config_root),
            output_dir=str(bundle.config_dir),
            wave=args.wave,
            experiment_ids=set(args.experiment_id) if args.experiment_id else None,
            include_deferred=False,
        )
        venv_activate = _resolve_bundle_venv(repository, args.venv_activate)
        venv_python = _venv_python_executable(venv_activate)
        write_submission_manifest(
            bundle,
            platform_id=profile.platform_id,
            wave=args.wave or "all",
            input_files={
                "platform_profile": source_profile,
                "execution_matrix": source_matrix,
                "slurm_profile": effective_profile,
            },
            environment_contract=capture_environment_contract(
                python_executable=venv_python,
                isolate_python_environment=venv_activate.resolve() != (repository / ".venv/bin/activate").resolve(),
            ),
        )
        manifest_hash = manifest_sha256(bundle.manifest_path)
        base_options = _resolve_slurm_options(
            profile_path=str(effective_profile),
            partition=args.partition or profile.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
        )
        setup_commands = _bundle_setup_commands(
            base_options["setup_commands"],
            bundle=bundle,
            venv_activate=venv_activate,
        )
        environment = {
            "HPC_AI_SUBMISSION_BUNDLE_ID": bundle.bundle_id,
            "HPC_AI_SUBMISSION_BUNDLE_MANIFEST_SHA256": manifest_hash,
            "HPC_AI_SUBMISSION_BUNDLE_PLATFORM": profile.platform_id,
            "HPC_AI_SUBMISSION_BUNDLE_WAVE": str(args.wave or "all"),
            "HPC_AI_ARTIFACT_OUTPUT_DIR": str(bundle.cache_dir / "runtime"),
        }
        _, script_paths = _render_platform_slurm(
            platform_profile=str(source_profile),
            execution_matrix=str(source_matrix),
            config_root=str(source_config_root),
            config_output_dir=str(bundle.config_dir),
            output_dir=str(bundle.script_dir),
            run_output_dir=str(bundle.result_dir),
            log_dir=str(bundle.log_dir),
            wave=args.wave,
            experiment_ids=set(args.experiment_id) if args.experiment_id else None,
            profile_path=str(effective_profile),
            partition=args.partition,
            account=args.account,
            time_limit=args.time,
            cpus_per_task=args.cpus_per_task,
            modules=args.module,
            env_values=args.env,
            setup_commands_override=setup_commands,
            environment_overrides=environment,
            submission_manifest=bundle.manifest_path,
            submission_manifest_sha256=manifest_hash,
            reuse_resolved_configs=True,
        )
        if sorted(config_paths) != sorted(bundle.config_dir.glob("E*.json")):
            raise RuntimeError("Submission bundle config set changed while rendering Slurm scripts")
        write_script_checksums(bundle)
        write_submission_plan(
            bundle.script_dir,
            bundle.root / "SUBMISSION_PLAN.md",
            batch_name=f"{profile.platform_id} wave {args.wave or 'all'} ({bundle.bundle_id})",
        )
        return bundle
    except Exception:
        remove_submission_bundle(bundle)
        raise


def _bundle_snapshot_path(bundle, source_path: str) -> Path:
    """Map one repository-relative input to the exact detached snapshot counterpart."""
    source = Path(source_path)
    repository = Path.cwd().resolve()
    resolved = source.resolve() if source.is_absolute() else (repository / source).resolve()
    try:
        relative = resolved.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"Submission bundle inputs must be inside the repository: {source_path}") from exc
    snapshot_path = bundle.repo_dir / relative
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Submission snapshot is missing requested input: {relative}")
    return snapshot_path


def _numeric_predecessor(value: str) -> str:
    if value and not value.isdigit():
        raise argparse.ArgumentTypeError("Slurm predecessor must be a numeric job ID")
    return value


def _submit_platform_wave(args: argparse.Namespace):
    """Create a bundle, persist its location, and submit its scripts in one action."""
    if not args.wave:
        raise ValueError("submit-platform-wave requires --wave")
    profile = load_platform_profile(args.platform_profile)
    if args.wave.startswith("inference-crossover-"):
        from hpc_ai_perf.config.inference_crossover import CANARY_WAVE
        from hpc_ai_perf.validation.inference_crossover import verify_inference_canaries

        if not args.venv_activate:
            raise ValueError("Inference crossover requires an explicit certified vLLM VENV_ACTIVATE")
        if args.wave != CANARY_WAVE:
            verify_inference_canaries(Path("results/raw") / profile.platform_id, profile.platform_id)
    if args.wave == "scaling-batch-crossover":
        from hpc_ai_perf.validation.scaling_acceptance import verify_scaling_canaries

        verify_scaling_canaries(
            Path("results/raw") / profile.platform_id,
            profile.platform_id,
            execution_matrix=args.execution_matrix,
        )
    if args.wave.startswith("scaling-dd-") and args.wave != "scaling-dd-canary":
        from hpc_ai_perf.validation.scaling_deep_dive import verify_deep_dive_canaries

        verify_deep_dive_canaries(
            Path("results/raw") / profile.platform_id,
            profile.platform_id,
            execution_matrix=args.execution_matrix,
        )
    if args.wave in {"qwen7b-update-controls-2gpu", "qwen7b-update-controls-4gpu"}:
        from hpc_ai_perf.validation.qwen7b_training import verify_update_canaries

        if profile.platform_id != "h100":
            raise ValueError("Qwen update controls are H100-only")
        verify_update_canaries(
            Path(os.environ.get("HPC_AI_QWEN_UPDATE_RESULT_ROOT", "results/raw/h100")),
            execution_matrix=args.execution_matrix,
        )
    if args.wave in {"lowbit-serving", "lowbit-bnb", "lowbit-quality"}:
        from hpc_ai_perf.validation.lowbit import probe_packages, verify_acceptance

        acceptance_dir = os.environ.get("HPC_AI_LOWBIT_ACCEPTANCE_DIR")
        if not acceptance_dir or not args.venv_activate:
            raise ValueError("Low-bit waves require make submit-lowbit-wave and a platform-specific acceptance job")
        runtime_python = str(Path(args.venv_activate).expanduser().parent / "python")
        verify_acceptance(
            Path(acceptance_dir),
            profile.platform_id,
            probe_packages(runtime_python),
            include_bnb=args.wave == "lowbit-bnb",
            wave=args.wave,
        )
    if args.wave == "vllm-long-context":
        from hpc_ai_perf.validation.long_context import verify_acceptance
        from hpc_ai_perf.validation.lowbit import probe_packages

        acceptance_dir = os.environ.get("HPC_AI_LONG_CONTEXT_ACCEPTANCE_DIR")
        if not acceptance_dir or not args.venv_activate:
            raise ValueError(
                "Long-context wave requires make submit-long-context-wave and a platform-specific acceptance job"
            )
        runtime_python = str(Path(args.venv_activate).expanduser().parent / "python")
        verify_acceptance(Path(acceptance_dir), profile.platform_id, probe_packages(runtime_python))
    if args.wave == "qwen7b-training":
        if profile.platform_id != "h100":
            raise ValueError("Qwen2.5-7B training is currently H100-only")
        from hpc_ai_perf.validation.qwen7b_training import verify_canaries

        verify_canaries(Path("results/raw/h100"), profile.platform_id)
    state_path = _wave_state_path_for_args(args, profile.platform_id)
    bundle = _create_platform_submission_bundle(args)
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    config_entries = manifest.get("configs") if isinstance(manifest, dict) else None
    if not isinstance(config_entries, dict):
        raise ValueError(f"Created bundle has invalid config manifest: {bundle.manifest_path}")
    status_path = write_job_status_csv(
        bundle.script_dir,
        bundle.root / "SLURM_STATUS.csv",
        batch_name=f"{profile.platform_id} wave {args.wave} ({bundle.bundle_id})",
    )
    append_wave_attempt(
        state_path,
        platform_id=profile.platform_id,
        wave=str(args.wave),
        bundle_dir=bundle.root,
        bundle_id=bundle.bundle_id,
        manifest_sha256=manifest_sha256(bundle.manifest_path),
        status_path=status_path,
        experiment_ids=sorted(str(experiment_id) for experiment_id in config_entries),
    )
    submissions = submit_job_scripts(bundle.script_dir, status_path, after_job=getattr(args, "after_job", ""))
    return bundle, state_path, submissions


def _current_platform_wave_status(args: argparse.Namespace):
    """Resolve the latest recorded attempt and its evidence coverage summary."""
    profile = load_platform_profile(args.platform_profile)
    state_path = _wave_state_path_for_args(args, profile.platform_id)
    attempt = current_wave_attempt(state_path)
    status = _apply_current_wave_policy(
        summarize_bundle_results(attempt["bundle_dir"]),
        profile.platform_id,
        str(args.wave),
    )
    return state_path, attempt, status


def _apply_current_wave_policy(
    status: BundleResultStatus,
    platform_id: str,
    wave: str,
) -> BundleResultStatus:
    """Show current planned scope while retaining coverage from an older bundle.

    A measured failure can move a condition from a runnable wave into deferred
    policy after its bundle was submitted. Status must stop recommending that
    historical condition for recovery, while still showing newly added policy
    conditions as missing until they receive their own submission.
    """
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        platform_id,
        wave=wave,
        include_deferred=False,
    )
    expected = {condition.matrix_id: condition.required_trials for condition in conditions}
    observed = {experiment.experiment_id: experiment for experiment in status.experiments}
    experiments: list[ExperimentResultStatus] = []
    for experiment_id, required_trials in sorted(expected.items()):
        experiment = observed.get(experiment_id)
        if experiment is None:
            experiment = ExperimentResultStatus(experiment_id, required_trials, (), (), 0)
        experiments.append(experiment)
    return BundleResultStatus(status.bundle_dir, tuple(experiments))


def _collect_platform_wave(args: argparse.Namespace) -> list[str]:
    """Import current-policy evidence, optionally preserving partial trial coverage."""
    profile = load_platform_profile(args.platform_profile)
    state_path = _wave_state_path_for_args(args, profile.platform_id)
    state = load_wave_state(state_path)
    expected_ids = _planned_wave_experiment_ids(profile.platform_id, str(args.wave))
    if not expected_ids:
        raise ValueError(f"No planned {profile.platform_id} conditions remain in wave {args.wave}")
    attempts = state["attempts"]
    summaries: list[tuple[dict[str, Any], Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError(f"Platform-wave state contains an invalid attempt: {state_path}")
        summaries.append((attempt, summarize_bundle_results(attempt["bundle_dir"])))

    completed_evidence_ids = _completed_planned_wave_evidence_ids(
        profile.platform_id,
        str(args.wave),
        result_dir=args.result_root,
    )
    selected: dict[str, tuple[dict[str, Any], Any]] = {}
    for attempt, status in summaries:
        for experiment in status.experiments:
            if experiment.experiment_id in expected_ids and experiment.complete:
                selected[experiment.experiment_id] = (attempt, experiment)
    missing = sorted(expected_ids - completed_evidence_ids - set(selected))
    allow_incomplete = getattr(args, "allow_incomplete", False)
    if missing and not allow_incomplete:
        raise ValueError(
            "Wave evidence is incomplete for: "
            + ", ".join(missing)
            + "; use --allow-incomplete to preserve completed trials or retry-platform-wave to recover"
        )

    imported_paths: list[str] = []
    result_root = Path(args.result_root)
    repository = Path.cwd().resolve()
    if allow_incomplete:
        sources = sorted(
            {
                path
                for _, status in summaries
                for experiment in status.experiments
                if experiment.experiment_id in expected_ids
                for path in experiment.completed_result_paths
            }
        )
    else:
        sources = [
            Path(selected[experiment_id][0]["bundle_dir"]) / "results" / experiment_id
            for experiment_id in sorted(selected)
        ]
    for source in sources:
        summary = import_result_evidence(source, result_root)
        for path in (*summary.imported, *summary.unchanged):
            resolved = path.resolve()
            try:
                imported_paths.append(str(resolved.relative_to(repository)))
            except ValueError as exc:
                raise ValueError(f"Collected evidence must remain inside the repository: {resolved}") from exc
    problems = verify_result_evidence(result_root)
    if problems:
        raise ValueError("Evidence verification failed after collection: " + "; ".join(problems))
    track_platform_progress(args.platform_profile, result_dir=result_root)
    pending = sorted(
        expected_ids - _completed_planned_wave_evidence_ids(profile.platform_id, str(args.wave), result_dir=result_root)
    )
    mark_wave_collected(state_path, imported_paths, incomplete_experiment_ids=pending)
    return sorted(set(imported_paths))


def _incomplete_planned_wave_experiment_ids(
    platform_id: str,
    wave: str,
    state_path: str | Path,
    *,
    result_dir: str | Path = "results/raw",
) -> list[str]:
    """Return planned conditions without accepted evidence or a complete saved attempt."""
    planned_ids = _planned_wave_experiment_ids(platform_id, wave)
    complete_ids = _completed_planned_wave_evidence_ids(platform_id, wave, result_dir=result_dir)
    if Path(state_path).is_file():
        state = load_wave_state(state_path)
        for attempt in state["attempts"]:
            if not isinstance(attempt, dict):
                raise ValueError(f"Platform-wave state contains an invalid attempt: {state_path}")
            status = summarize_bundle_results(attempt["bundle_dir"])
            for experiment in status.experiments:
                if experiment.experiment_id in planned_ids and experiment.complete:
                    complete_ids.add(experiment.experiment_id)
    return sorted(planned_ids - complete_ids)


def _completed_planned_wave_evidence_ids(
    platform_id: str,
    wave: str,
    *,
    result_dir: str | Path = "results/raw",
) -> set[str]:
    """Return conditions already satisfied by validated evidence in the accepted result root."""
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        platform_id,
        wave=wave,
        include_deferred=False,
    )
    rows = build_platform_progress_rows(conditions, result_dir)
    return {str(row["matrix_id"]) for row in rows if row["status"] == "Complete"}


def _planned_wave_experiment_ids(platform_id: str, wave: str) -> set[str]:
    """Return the present policy scope for collection, excluding newly deferred rows.

    A failed attempt can cause a condition to be explicitly deferred after
    evidence shows it is unsupported on a platform. Collection must not keep
    requiring that historical condition once a newer matrix policy has removed
    it from the runnable wave; completed trials from earlier compatible
    attempts remain eligible.
    """
    conditions = load_platform_conditions(
        "platforms/execution_matrix.json",
        platform_id,
        wave=wave,
        include_deferred=False,
    )
    return {condition.matrix_id for condition in conditions if condition.status == "planned"}


def _stage_platform_wave(args: argparse.Namespace) -> list[str]:
    """Stage exactly the collected raw evidence and its checksum manifest."""
    profile = load_platform_profile(args.platform_profile)
    state_path = _wave_state_path_for_args(args, profile.platform_id)
    state = load_wave_state(state_path)
    imported_paths = state.get("imported_paths")
    if not isinstance(imported_paths, list) or not imported_paths:
        raise ValueError("No collected evidence is recorded; run collect-platform-wave first")
    platform_evidence_root = Path("results") / "raw" / profile.platform_id
    checksum = platform_evidence_root / "SHA256SUMS"
    # SHA256SUMS covers the full platform tree. Stage the same complete set so
    # a commit can never contain a manifest that refers to an omitted trial.
    paths = sorted(str(path) for path in platform_evidence_root.rglob("*.json")) + [str(checksum)]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Collected evidence is missing: " + ", ".join(missing))
    subprocess.run(["git", "add", "--", *paths], check=True)
    return paths


def _wave_state_path_for_args(args: argparse.Namespace, platform_id: str) -> Path:
    wave = str(args.wave)
    if not wave or wave == "None":
        raise ValueError("Platform-wave commands require --wave")
    return Path(args.state_file) if args.state_file else wave_state_path(Path.cwd(), platform_id, wave)


def _summarize_scheduler_rows(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        state = row.get("state", "UNKNOWN") or "UNKNOWN"
        counts[state] = counts.get(state, 0) + 1
    return ", ".join(f"{state}={count}" for state, count in sorted(counts.items())) or "no jobs recorded"


def _resolve_bundle_venv(repository: Path, requested_path: str | None) -> Path:
    path = Path(requested_path).expanduser() if requested_path else repository / ".venv" / "bin" / "activate"
    resolved = path.resolve() if path.is_absolute() else (repository / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Virtualenv activation script is missing: {resolved}. Use --venv-activate to select the shared runtime."
        )
    return resolved


def _venv_python_executable(venv_activate: Path) -> Path:
    """Return the virtualenv launcher selected by a standard ``bin/activate`` script."""
    python_executable = venv_activate.parent / "python"
    if not python_executable.is_file():
        raise FileNotFoundError(
            f"Virtualenv Python executable is missing: {python_executable}. "
            "Recreate the environment or select a valid --venv-activate path."
        )
    # Keep the launcher path rather than following it to the base interpreter.
    # Python discovers ``pyvenv.cfg`` from this path; following the symlink
    # would cause bundle contract capture to run outside the selected venv.
    return python_executable


def _bundle_setup_commands(
    base_commands: list[str],
    *,
    bundle,
    venv_activate: Path,
) -> list[str]:
    """Run cluster setup in the snapshot while activating the explicitly shared virtualenv."""
    project_venv_activate = (bundle.source_repository / ".venv" / "bin" / "activate").resolve()
    isolate_python_environment = venv_activate.resolve() != project_venv_activate
    commands = [
        f"cd {quote(str(bundle.repo_dir))}",
        f"export TORCH_EXTENSIONS_DIR={quote(str(bundle.cache_dir / 'torch_extensions'))}",
    ]
    activated = False
    for command in base_commands:
        if command.strip() == "source .venv/bin/activate":
            if isolate_python_environment:
                # Module stacks commonly inject their Python packages through
                # PYTHONPATH. Preserve that behaviour for the project .venv,
                # which intentionally consumes site packages, but clear it
                # before an isolated serving environment so module Torch cannot
                # shadow its private Gloo/NCCL-enabled build.
                commands.append("unset PYTHONPATH PYTHONHOME")
            commands.append(f"source {quote(str(venv_activate))}")
            activated = True
        else:
            commands.append(command)
    if not activated:
        if isolate_python_environment:
            commands.append("unset PYTHONPATH PYTHONHOME")
        commands.append(f"source {quote(str(venv_activate))}")
    source_path = quote(str(bundle.repo_dir / "src"))
    if isolate_python_environment:
        commands.append(f"export PYTHONPATH={source_path}")
    else:
        commands.append(f"export PYTHONPATH={source_path}${{PYTHONPATH:+:$PYTHONPATH}}")
    return commands


def _git_head(repository: Path) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _refresh_outputs(
    *,
    config_root: str,
    result_dir: str,
    processed_results: str,
    analysis_dir: str,
    tracker_csv: str,
    tracker_markdown: str,
    readiness_report: str,
    skip_validation: bool,
) -> None:
    """Rebuild the standard derived project outputs from configs and raw results.

    The refresh command is the normal "make the repository reports current"
    operation. It validates first unless explicitly skipped, aggregates raw JSON
    records into the processed CSV, rebuilds analysis tables, refreshes the
    matrix progress tracker, and writes the readiness report used before HPC
    submission. Dev tracking is intentionally separate so local Mac results do
    not change the status of the dissertation matrix.
    """
    if not skip_validation:
        issues = validate_project(config_root, result_dir)
        if issues:
            print(format_validation_issues(issues), end="")
            raise SystemExit(1)
        LOGGER.info("Validation passed.")

    result_path = aggregate_results(result_dir, processed_results)
    LOGGER.info("Wrote aggregated results to %s", result_path)

    analysis_paths = analyze_results(result_path, analysis_dir)
    LOGGER.info("Wrote %d analysis tables to %s", len(analysis_paths), analysis_dir)

    telemetry_csv, telemetry_markdown = audit_telemetry(
        result_dir,
        Path(analysis_dir) / "telemetry_audit.csv",
        Path(analysis_dir) / "telemetry_audit.md",
    )
    LOGGER.info("Wrote telemetry audit to %s and %s", telemetry_csv, telemetry_markdown)

    # Keep refresh focused on the real E-number matrix tracker; dev has its own Make target.
    tracker_csv_path, tracker_markdown_path = track_experiment_progress(
        config_root,
        result_dir,
        tracker_csv,
        tracker_markdown,
    )
    LOGGER.info("Wrote tracker CSV to %s", tracker_csv_path)
    LOGGER.info("Wrote tracker Markdown to %s", tracker_markdown_path)

    readiness_path = write_readiness_report(
        readiness_report,
        config_root=config_root,
        result_dir=result_dir,
    )
    LOGGER.info("Wrote readiness report to %s", readiness_path)


def _render_slurm_config(
    config_path: str,
    *,
    output_dir: str,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
    profile_path: str | None,
    print_script: bool,
    repetitions: int = 1,
) -> None:
    """Render one Slurm script for one concrete experiment config.

    CLI-provided scheduler values override profile defaults. The resolved
    values are passed to the Slurm renderer along with the loaded experiment
    config, and the rendered script is either printed for inspection or written
    to the requested output directory.
    """
    slurm_options = _resolve_slurm_options(
        profile_path=profile_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        env_values=env_values,
    )
    config = load_experiment_config(config_path)
    semantic_issues = capability_issues(config)
    if semantic_issues:
        raise ValueError(f"{config.experiment_id} is not research-ready: {'; '.join(semantic_issues)}")
    _verify_profile_platform(config, slurm_options.get("platform"), profile_path)
    script = render_slurm_script(
        config,
        config_path,
        partition=slurm_options["partition"],
        account=slurm_options["account"],
        time_limit=slurm_options["time_limit"],
        cpus_per_task=slurm_options["cpus_per_task"],
        modules=slurm_options["modules"],
        environment=slurm_options["environment"],
        setup_commands=slurm_options["setup_commands"],
        repetitions=repetitions,
    )
    if print_script:
        print(script, end="")
        return

    script_path = write_slurm_script(script, output_dir, config.experiment_id)
    LOGGER.info("Wrote Slurm script to %s", script_path)


def _render_slurm_bulk(
    *,
    config_root: str,
    output_dir: str,
    phase: str | None,
    priority: str | None,
    mode: str | None,
    workload: str | None,
    experiment_id: str | None,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
    profile_path: str | None,
    repetitions: int = 1,
) -> None:
    """Render Slurm scripts for every matrix config matching the supplied filters.

    Bulk rendering is deliberately matrix-only because generated HPC batches
    should correspond to real E-number dissertation experiments. The filters
    make it possible to render conservative batches, such as Must Have
    environment-validation jobs, without touching the rest of the matrix.
    """
    entries = discover_experiment_configs(config_root)
    entries = filter_config_entries(
        entries,
        phase=phase,
        priority=priority,
        mode=mode,
        workload=workload,
        experiment_id=experiment_id,
    )
    if not entries:
        LOGGER.warning("No configs matched the requested filters.")
        return

    deferred = [entry for entry in entries if entry.execution_status.lower() == "deferred"]
    entries = [entry for entry in entries if entry.execution_status.lower() != "deferred"]
    if deferred:
        LOGGER.info(
            "Skipping %d deferred config(s) outside executable scope: %s",
            len(deferred),
            ", ".join(entry.config.experiment_id for entry in deferred),
        )
    if not entries:
        LOGGER.warning("All matching configs are deferred; no Slurm scripts generated.")
        return

    slurm_options = _resolve_slurm_options(
        profile_path=profile_path,
        partition=partition,
        account=account,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        modules=modules,
        env_values=env_values,
    )
    for entry in entries:
        semantic_issues = capability_issues(entry.config)
        if semantic_issues:
            raise ValueError(f"{entry.config.experiment_id} is not research-ready: {'; '.join(semantic_issues)}")
        _verify_profile_platform(entry.config, slurm_options.get("platform"), profile_path)
        script = render_slurm_script(
            entry.config,
            entry.path,
            partition=slurm_options["partition"],
            account=slurm_options["account"],
            time_limit=slurm_options["time_limit"],
            cpus_per_task=slurm_options["cpus_per_task"],
            modules=slurm_options["modules"],
            environment=slurm_options["environment"],
            setup_commands=slurm_options["setup_commands"],
            repetitions=repetitions,
        )
        write_slurm_script(script, output_dir, entry.config.experiment_id)

    LOGGER.info("Wrote %d Slurm scripts to %s", len(entries), output_dir)


def _parse_environment(values: list[str]) -> dict[str, str]:
    """Parse repeated `--env KEY=VALUE` options into a Slurm environment mapping.

    The parser fails early for malformed values because an invalid environment
    assignment would otherwise become a confusing scheduler or shell error after
    the job has already been submitted.
    """
    environment: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected environment value in KEY=VALUE form: {value}")
        key, env_value = value.split("=", 1)
        if not key:
            raise ValueError(f"Environment key cannot be empty: {value}")
        environment[key] = env_value
    return environment


def _resolve_slurm_options(
    *,
    profile_path: str | None,
    partition: str | None,
    account: str | None,
    time_limit: str | None,
    cpus_per_task: int | None,
    modules: list[str],
    env_values: list[str],
) -> dict[str, Any]:
    """Merge Slurm profile defaults with explicit command-line overrides.

    Profiles capture stable site defaults such as the partition, setup commands,
    module list, and environment variables. Direct CLI flags take precedence for
    fields that are often adjusted per run, while modules and environment values
    are additive so the caller can extend a profile without rewriting it.
    """
    profile = _load_slurm_profile(profile_path) if profile_path else {}
    profile_environment = profile.get("environment", {})

    return {
        "partition": partition or profile.get("partition"),
        "account": account or profile.get("account"),
        "time_limit": time_limit or profile.get("time"),
        "cpus_per_task": cpus_per_task or profile.get("cpus_per_task", 8),
        "modules": [*profile.get("modules", []), *modules],
        "environment": {**profile_environment, **_parse_environment(env_values)},
        "setup_commands": profile.get("setup_commands", []),
        "platform": profile.get("platform"),
    }


def _load_slurm_profile(profile_path: str) -> dict[str, Any]:
    """Load and type-check a JSON Slurm profile.

    This validation is intentionally structural rather than cluster-aware: it
    confirms the profile uses supported keys and value types, while
    `_check_slurm_profile` performs readiness checks for placeholders and
    missing operational settings.
    """
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"Slurm profile must be a JSON object: {profile_path}")

    unknown_keys = sorted(set(profile) - SLURM_PROFILE_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown Slurm profile keys in {profile_path}: {', '.join(unknown_keys)}")

    _validate_string_or_none(profile, "partition", profile_path)
    _validate_string_or_none(profile, "platform", profile_path)
    _validate_string_or_none(profile, "account", profile_path)
    _validate_string_or_none(profile, "time", profile_path)
    _validate_int_or_none(profile, "cpus_per_task", profile_path)
    _validate_string_list(profile, "modules", profile_path)
    _validate_string_list(profile, "setup_commands", profile_path)
    _validate_string_dict(profile, "environment", profile_path)
    return profile


def _check_slurm_profile(profile_path: str) -> list[str]:
    """Return human-readable readiness issues for a Slurm profile.

    A profile can be structurally valid JSON but still unsafe to use on a real
    cluster if it contains placeholder values or no environment setup. This
    helper performs that higher-level preflight check before rendering or
    submitting HPC scripts.
    """
    profile = _load_slurm_profile(profile_path)
    issues: list[str] = []

    partition = profile.get("partition")
    if not partition:
        issues.append("`partition` must be set.")
    elif _contains_placeholder(partition):
        issues.append(f"`partition` still contains a placeholder: {partition}")

    platform_name = profile.get("platform")
    if not platform_name:
        issues.append("`platform` must identify the target hardware, such as lovelace_h100 or lovelace_l40s.")
    elif _contains_placeholder(platform_name):
        issues.append(f"`platform` still contains a placeholder: {platform_name}")

    account = profile.get("account")
    if account and _contains_placeholder(account):
        issues.append(f"`account` still contains a placeholder: {account}")

    modules = profile.get("modules", [])
    setup_commands = profile.get("setup_commands", [])
    if not modules and not setup_commands:
        issues.append("`modules` or `setup_commands` should load the CUDA/Python stack required on the cluster.")
    for index, command in enumerate(setup_commands, start=1):
        if _contains_placeholder(command):
            issues.append(f"`setup_commands[{index}]` still contains a placeholder: {command}")

    environment = profile.get("environment", {})
    for key, value in environment.items():
        if _contains_placeholder(value):
            issues.append(f"`environment.{key}` still contains a placeholder: {value}")

    return issues


def _verify_profile_platform(config: Any, profile_platform: str | None, profile_path: str | None) -> None:
    """Prevent hardware-labelled configs from using a profile for another GPU platform."""
    if not profile_path:
        return
    if not profile_platform:
        raise ValueError(f"Slurm profile {profile_path} must declare `platform` for research jobs")
    if config.hardware.platform.startswith("lovelace_") and config.hardware.platform != profile_platform:
        raise ValueError(
            f"{config.experiment_id} targets {config.hardware.platform}, but {profile_path} targets {profile_platform}"
        )


def _contains_placeholder(value: str) -> bool:
    """Detect common placeholder tokens that should not reach real sbatch files."""
    return any(token.lower() in value.lower() for token in SLURM_PROFILE_PLACEHOLDERS)


def _validate_string_or_none(profile: dict[str, Any], key: str, profile_path: str) -> None:
    """Require a profile key to be absent, null, or a string."""
    if key in profile and profile[key] is not None and not isinstance(profile[key], str):
        raise ValueError(f"Slurm profile key '{key}' must be a string: {profile_path}")


def _validate_int_or_none(profile: dict[str, Any], key: str, profile_path: str) -> None:
    """Require a profile key to be absent, null, or an integer."""
    if key in profile and profile[key] is not None and not isinstance(profile[key], int):
        raise ValueError(f"Slurm profile key '{key}' must be an integer: {profile_path}")


def _validate_string_list(profile: dict[str, Any], key: str, profile_path: str) -> None:
    """Require a profile key to be a list containing only strings when present."""
    if key not in profile:
        return
    value = profile[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Slurm profile key '{key}' must be a list of strings: {profile_path}")


def _validate_string_dict(profile: dict[str, Any], key: str, profile_path: str) -> None:
    """Require a profile key to be an object with string keys and string values."""
    if key not in profile:
        return
    value = profile[key]
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError(f"Slurm profile key '{key}' must be an object of string values: {profile_path}")


if __name__ == "__main__":
    main()
