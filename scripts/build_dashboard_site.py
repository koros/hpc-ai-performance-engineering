"""Assemble the static GitHub Pages artifact from refreshed dashboard outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DASHBOARD_FILES = ("index.html", "app.js", "styles.css")
ANALYSIS_FILES = (
    "communication_summary.csv",
    "data_movement_summary.csv",
    "experiment_rankings.csv",
    "hardware_summary.csv",
    "inference_summary.csv",
    "memory_summary.csv",
    "phase_summary.csv",
    "precision_summary.csv",
    "research_finding_summary.csv",
    "research_question_summary.csv",
    "scaling_summary.csv",
    "strategy_summary.csv",
    "trial_summary.csv",
)
PROCESSED_FILES = ("experiments.csv", "readiness_report.md")

ROOT_REDIRECT = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="0; url=dashboard/">
    <title>HPC AI Performance Engineering</title>
  </head>
  <body>
    <p><a href="dashboard/">Open the experiment dashboard</a></p>
  </body>
</html>
"""


def build_site(repository: Path, output_dir: Path) -> list[Path]:
    repository = repository.resolve()
    output_dir = output_dir.resolve()
    build_root = (repository / "build").resolve()
    if output_dir == build_root or build_root not in output_dir.parents:
        raise ValueError("Dashboard publication output must be a child of the repository build directory")

    sources = [repository / "dashboard" / name for name in DASHBOARD_FILES]
    sources.extend(repository / "results" / "analysis" / name for name in ANALYSIS_FILES)
    sources.extend(repository / "results" / "processed" / name for name in PROCESSED_FILES)
    missing = [path.relative_to(repository) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing refreshed dashboard input(s): " + ", ".join(map(str, missing)))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    dashboard_dir = output_dir / "dashboard"
    analysis_dir = output_dir / "results" / "analysis"
    processed_dir = output_dir / "results" / "processed"
    dashboard_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    copied: list[Path] = []
    for name in DASHBOARD_FILES:
        target = dashboard_dir / name
        shutil.copy2(repository / "dashboard" / name, target)
        copied.append(target)
    for name in ANALYSIS_FILES:
        target = analysis_dir / name
        shutil.copy2(repository / "results" / "analysis" / name, target)
        copied.append(target)
    for name in PROCESSED_FILES:
        target = processed_dir / name
        shutil.copy2(repository / "results" / "processed" / name, target)
        copied.append(target)

    (output_dir / "index.html").write_text(ROOT_REDIRECT, encoding="utf-8")
    (output_dir / ".nojekyll").touch()
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("build/pages"))
    args = parser.parse_args()
    copied = build_site(args.repository, args.output_dir)
    print(f"Built GitHub Pages artifact with {len(copied)} dashboard data files at {args.output_dir}")


if __name__ == "__main__":
    main()
