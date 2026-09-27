from pathlib import Path

import pytest
from scripts.build_dashboard_site import ANALYSIS_FILES, DASHBOARD_FILES, PROCESSED_FILES, build_site


def test_publication_refreshes_derived_inputs_before_dashboard_checks() -> None:
    workflow = Path(".github/workflows/dashboard-pages.yml").read_text(encoding="utf-8")
    publication = workflow.split("\n  publish:\n", maxsplit=1)[1]
    commands = [
        line.strip().removeprefix("run: ") for line in publication.splitlines() if line.strip().startswith("run: ")
    ]

    assert commands.index("make refresh PYTHON=python") < commands.index("make check PYTHON=python")
    assert commands.index("make check PYTHON=python") < commands.index("make build-dashboard-site PYTHON=python")


def test_build_site_publishes_only_dashboard_and_derived_evidence(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    for name in DASHBOARD_FILES:
        write(repository / "dashboard" / name, name)
    for name in ANALYSIS_FILES:
        write(repository / "results" / "analysis" / name, name)
    for name in PROCESSED_FILES:
        write(repository / "results" / "processed" / name, name)
    write(repository / "results" / "raw" / "h100" / "E001.json", "raw evidence")
    write(repository / "dashboard" / "app.test.cjs", "test code")

    output = repository / "build" / "pages"
    copied = build_site(repository, output)

    assert len(copied) == len(DASHBOARD_FILES) + len(ANALYSIS_FILES) + len(PROCESSED_FILES)
    assert (output / "index.html").is_file()
    assert (output / ".nojekyll").is_file()
    assert (output / "dashboard" / "app.js").is_file()
    assert (output / "dashboard" / "logo.svg").is_file()
    assert 'href="dashboard/logo.svg"' in (output / "index.html").read_text(encoding="utf-8")
    assert (output / "results" / "analysis" / "memory_summary.csv").is_file()
    assert not (output / "results" / "raw").exists()
    assert not (output / "dashboard" / "app.test.cjs").exists()


def test_build_site_rejects_destructive_output_outside_build(tmp_path: Path) -> None:
    repository = tmp_path / "repository"

    with pytest.raises(ValueError, match="child of the repository build directory"):
        build_site(repository, tmp_path / "elsewhere")


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
