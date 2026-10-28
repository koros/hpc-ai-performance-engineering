"""Reject personal home-directory prefixes in tracked files or public history."""

from __future__ import annotations

import argparse
import re
import subprocess

PERSONAL_PATH = re.compile(rb"/(?:users|home)/[a-z0-9_.-]+/", re.IGNORECASE)


def personal_path_lines(data: bytes) -> list[int]:
    return [number for number, line in enumerate(data.splitlines(), 1) if PERSONAL_PATH.search(line)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Scan objects reachable from branches and tags")
    args = parser.parse_args()
    findings = []
    if args.history:
        objects = subprocess.check_output(["git", "rev-list", "--objects", "--branches", "--tags"]).splitlines()
        process = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert process.stdin is not None and process.stdout is not None
        try:
            for entry in objects:
                oid, _, path = entry.partition(b" ")
                process.stdin.write(oid + b"\n")
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3:
                    raise RuntimeError("Cannot inspect Git object")
                data = process.stdout.read(int(header[2]))
                process.stdout.read(1)
                if header[1] in {b"blob", b"commit", b"tag"}:
                    for line in personal_path_lines(data):
                        findings.append(f"{oid.decode()}:{path.decode(errors='replace')}:{line}")
        finally:
            process.stdin.close()
            process.stdout.close()
            if process.wait() != 0:
                raise RuntimeError("Git object inspection failed")
    else:
        from pathlib import Path

        for name in subprocess.check_output(["git", "ls-files", "-z"]).split(b"\0"):
            if name:
                path = Path(name.decode())
                if path.is_file():
                    findings.extend(f"{path}:{line}" for line in personal_path_lines(path.read_bytes()))
    for finding in findings:
        print(f"Personal path prefix: {finding}")
    print(f"Publication path check: {len(findings)} finding(s).")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
