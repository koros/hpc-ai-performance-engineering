"""Assemble the self-contained experiment catalog and findings for GitHub Pages."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DASHBOARD_FILES = ("index.html", "analysis.html", "README.md", "experiment-details.json", "experiments-data.json")

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
    missing = [path.relative_to(repository) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing dashboard input(s): " + ", ".join(map(str, missing)))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    dashboard_dir = output_dir / "dashboard"
    dashboard_dir.mkdir(parents=True)

    copied: list[Path] = []
    for name in DASHBOARD_FILES:
        target = dashboard_dir / name
        shutil.copy2(repository / "dashboard" / name, target)
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
    print(f"Built GitHub Pages artifact with {len(copied)} dashboard files at {args.output_dir}")


if __name__ == "__main__":
    main()
