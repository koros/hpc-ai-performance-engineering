"""Read-only compatibility audit of a historical topology bundle versus current evidence.

This report does not certify binary equivalence or import/change raw evidence.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from hpc_ai_perf.validation import validate_result_file
from hpc_ai_perf.workloads.scaling import scaling_runtime_sha256

VERSION_KEYS = ("torch_version", "cuda_version", "cudnn_version", "nccl_version", "transformers_version")


def audit(bundle, current_root):
    paths = sorted((bundle / "results").glob("E*/*.json"))
    if not paths:
        raise ValueError(f"No historical result JSON found under {bundle / 'results'}")
    identities, evidence, counts = [], [], Counter()
    for path in paths:
        r = json.loads(path.read_text())
        problems = validate_result_file(path)
        status = "schema_invalid" if problems else r.get("status", "unknown")
        counts[status] += 1
        identity = dict(
            runtime_sha256=r.get("metrics", {}).get("scaling_runtime_sha256"),
            **{k: r.get("environment", {}).get(k) for k in VERSION_KEYS},
        )
        if identity not in identities:
            identities.append(identity)
        evidence.append(
            dict(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status=status,
                validation_issues=problems,
            )
        )
    current = []
    for path in sorted(current_root.glob("E79[0-5]/*.json")):
        r = json.loads(path.read_text())
        if r.get("status") == "completed" and not validate_result_file(path):
            current.append(
                dict(
                    path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    runtime_sha256=r.get("metrics", {}).get("scaling_runtime_sha256"),
                    **{k: r.get("environment", {}).get(k) for k in VERSION_KEYS},
                )
            )
    expected = scaling_runtime_sha256()
    return dict(
        historical_bundle=str(bundle),
        historical_results=dict(counts),
        historical_runtime_identities=identities,
        current_runtime_sha256=expected,
        historical_fingerprint_matches_current=all(i["runtime_sha256"] == expected for i in identities),
        current_comparison_evidence=current,
        historical_files=evidence,
        decision="separate recovery cohort with fresh matched controls; preserve historical successes",
        limitation="Version strings/fingerprints do not establish identical binary builds or library paths. "
        "Current comparison evidence is informative; acceptance must pass separately.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=Path("results/raw/l40s"))
    parser.add_argument("--output", type=Path, default=Path("build/topology_recovery/compatibility_audit.json"))
    args = parser.parse_args()
    report = audit(args.bundle, args.current_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Historical results:", report["historical_results"])
    print("Historical fingerprint matches current:", report["historical_fingerprint_matches_current"])
    print("Decision:", report["decision"])
    print("Audit:", args.output)


if __name__ == "__main__":
    main()
