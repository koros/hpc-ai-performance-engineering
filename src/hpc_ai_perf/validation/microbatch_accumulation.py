"""Small, study-specific acceptance gate; historical scaling canaries do not unlock it."""

import hashlib
import json
from pathlib import Path

from hpc_ai_perf.config import discover_experiment_configs
from hpc_ai_perf.config.microbatch_accumulation import CANARY_WAVE, CONFIG_ROOT, design
from hpc_ai_perf.config.platform import load_platform_conditions, load_platform_profile, resolve_platform_config
from hpc_ai_perf.validation import validate_result_file
from hpc_ai_perf.validation.qwen7b_training import canary_contract_sha256
from hpc_ai_perf.validation.scaling_acceptance import _manifest_entries
from hpc_ai_perf.validation.scaling_deep_dive import _canary_issues


def verify_microbatch_canaries(result_root, *, execution_matrix="platforms/execution_matrix.json"):
    root = Path(result_root)
    checksums = _manifest_entries(root / "SHA256SUMS")
    conditions = load_platform_conditions(execution_matrix, "l40s", wave=CANARY_WAVE)
    expected = {eid for eid, wave, _ in design() if wave == CANARY_WAVE}
    if {c.matrix_id for c in conditions} != expected or any(c.required_trials != 1 for c in conditions):
        raise ValueError("Microbatch canary policy must contain all six one-trial conditions")
    entries = {e.matrix_id: e for e in discover_experiment_configs(CONFIG_ROOT)}
    profile = load_platform_profile("platforms/l40s.json")
    accepted, issues, identities, digests = {}, [], set(), {}
    for condition in conditions:
        eid = condition.matrix_id
        config = resolve_platform_config(entries[eid], profile, condition)
        reasons = set()
        # Most recent matching attempt controls acceptance; no fallback to older successful evidence.
        candidates = []
        for path in (root / eid).glob("*.json"):
            try:
                r = json.loads(path.read_text())
            except (OSError, ValueError):
                reasons.add("unreadable JSON")
                continue
            if (
                isinstance(r, dict)
                and isinstance(r.get("config"), dict)
                and canary_contract_sha256(r["config"]) == canary_contract_sha256(config)
            ):
                candidates.append((str(r.get("started_at", "")), str(path), r))
        if candidates:
            _, filename, record = max(candidates, key=lambda item: item[:2])
            path = Path(filename)
            if checksums.get(path.relative_to(root).as_posix()) != hashlib.sha256(path.read_bytes()).hexdigest():
                reasons.add("checksum")
            if validate_result_file(path):
                reasons.add("evidence schema")
            reasons.update(_canary_issues(record, config, condition.condition_id, "l40s"))
            if not reasons:
                env = record.get("environment", {})
                identity = tuple(env.get(k) for k in ("torch_version", "cuda_version", "transformers_version"))
                if any(not v for v in identity):
                    reasons.add("missing runtime versions")
                else:
                    identities.add(identity)
                    digests.setdefault(config.hardware.gpus, set()).add(record["metrics"]["global_input_sha256"])
                    accepted[eid] = filename
        if eid not in accepted:
            issues.append(f"{eid}: " + (", ".join(sorted(reasons)) or "no matching collected canary"))
    if len(identities) > 1:
        issues.append("runtime versions differ across canaries")
    if any(len(values) != 1 for values in digests.values()):
        issues.append("global inputs differ between microbatch splits at the same GPU count")
    if issues:
        raise ValueError("Microbatch accumulation acceptance failed:\n- " + "\n- ".join(issues))
    return accepted
