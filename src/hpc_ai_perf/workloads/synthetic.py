from __future__ import annotations

import time

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


class SyntheticWorkload(Workload):
    """Small CPU-only workload for validating the local execution path."""

    name = "synthetic"

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        steps = config.runtime.max_steps or 5
        tokens_per_step = int(config.workload.parameters.get("tokens_per_step", 1024))
        sleep_seconds = float(config.workload.parameters.get("sleep_seconds", 0.01))

        start = time.perf_counter()
        for _ in range(steps):
            time.sleep(sleep_seconds)
        elapsed = max(time.perf_counter() - start, 1e-9)

        total_tokens = steps * tokens_per_step
        return WorkloadRunOutput(
            metrics={
                "steps": steps,
                "total_tokens": total_tokens,
                "throughput_tokens_sec": total_tokens / elapsed,
            }
        )

