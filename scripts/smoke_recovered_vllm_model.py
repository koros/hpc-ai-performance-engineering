#!/usr/bin/env python3
"""Two-GPU model/timestamp diagnostic; separate from benchmark canaries."""

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "benchmark_evidence": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.executable,
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        import torch
        import vllm
        from vllm import LLM, SamplingParams

        if torch.cuda.device_count() != 2:
            raise RuntimeError("Exactly two allocated GPUs required")
        if os.environ.get("VLLM_USE_V1") != "0":
            raise RuntimeError("Timing check requires explicit VLLM_USE_V1=0")
        report.update(
            torch_version=torch.__version__,
            vllm_version=vllm.__version__,
            gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
        )
        engine = LLM(
            model=report["model"],
            revision=report["revision"],
            tensor_parallel_size=2,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=2048,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.85,
        )
        sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
        prompts = [{"prompt_token_ids": [42] * 32}, {"prompt_token_ids": [43] * 32}]
        engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
        outputs = engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
        if len(outputs) != 2:
            raise RuntimeError("Expected two completed requests")
        timings = []
        import math

        for output in outputs:
            if not output.finished or len(output.outputs) != 1 or len(output.outputs[0].token_ids) != 16:
                raise RuntimeError("Incomplete generation or wrong output-token count")
            m = output.metrics
            values = [
                getattr(m, key, None)
                for key in ("arrival_time", "first_token_time", "last_token_time", "finished_time")
            ]
            if (
                any(v is None or not math.isfinite(v) for v in values)
                or not values[0] < values[1] < values[2] <= values[3]
            ):
                raise RuntimeError(f"Missing or invalid request timestamps: {values}")
            timings.append(dict(zip(("arrival", "first_token", "last_token", "finished"), values, strict=True)))
        report.update(status="passed", request_timestamps=timings, requests=2, output_tokens_per_request=16)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Model/timestamp recovery diagnostic passed: {args.output}")


if __name__ == "__main__":
    main()
