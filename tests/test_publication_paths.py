from __future__ import annotations

import importlib.util


def test_personal_path_scan_covers_home_prefix_case_variants():
    spec = importlib.util.spec_from_file_location("publication_paths", "scripts/check_publication_paths.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for prefix in ("users", "Users", "USERS", "home"):
        path = "/".join(("", prefix, "example", "Workspace", "project"))
        assert module.personal_path_lines(("safe\n" + path).encode()) == [2]
    assert module.personal_path_lines(b"../hpc-ai-submissions/bundle\n./configs/example.json\n${HOME}/.local") == []
