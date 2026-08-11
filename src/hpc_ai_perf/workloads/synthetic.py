from __future__ import annotations

import time

from hpc_ai_perf.config import ExperimentConfig
from hpc_ai_perf.workloads.base import Workload, WorkloadRunOutput


class SyntheticWorkload(Workload):
    """Small CPU-only workload for validating the local execution path.

    The synthetic workload is deliberately simple: it does not import PyTorch,
    download model weights, require a GPU, or depend on external datasets. That
    makes it useful as the first smoke check for the runner, result writer,
    validation logic, aggregation, and dashboard plumbing.
    """

    name = "synthetic"

    def run(self, config: ExperimentConfig) -> WorkloadRunOutput:
        """Run a deterministic timing loop and report token-style throughput.

        `max_steps` controls how many loop iterations are executed.
        `tokens_per_step` gives the loop a benchmark-like unit of work even
        though no real model is trained. `sleep_seconds` keeps the workload cheap
        while still producing non-zero elapsed time, which is useful for testing
        throughput calculations and result schemas.
        """
        steps = config.runtime.max_steps or 5
        tokens_per_step = int(config.workload.parameters.get("tokens_per_step", 1024))
        sleep_seconds = float(config.workload.parameters.get("sleep_seconds", 0.01))

        # Use a monotonic high-resolution timer so very short local smoke runs
        # produce stable elapsed-time measurements independent of wall-clock time.
        start = time.perf_counter()
        for _ in range(steps):
            time.sleep(sleep_seconds)
        elapsed = max(time.perf_counter() - start, 1e-9)

        # Report the same token-throughput shape used by transformer workloads
        # so aggregation and dashboard paths can be tested with no ML dependency.
        total_tokens = steps * tokens_per_step
        return WorkloadRunOutput(
            metrics={
                "steps": steps,
                "total_tokens": total_tokens,
                "throughput_tokens_sec": total_tokens / elapsed,
            }
        )
