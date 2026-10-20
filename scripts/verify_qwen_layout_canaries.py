"""Validate model-specific Qwen7B equal-budget acceptance evidence."""

import argparse

from hpc_ai_perf.config.qwen_layout import CANARY_WAVE, CONFIG_ROOT, EXTENSION_CANARY_WAVE, LOW_CONCURRENCY_CANARY_WAVE
from hpc_ai_perf.validation.inference_crossover import verify_inference_canaries

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extension", action="store_true", help="Require original plus TP4 acceptance")
parser.add_argument("--low-concurrency", action="store_true", help="Also require all concurrency-one canaries")
args = parser.parse_args()
accepted = verify_inference_canaries("results/raw/l40s", "l40s", config_root=CONFIG_ROOT, canary_wave=CANARY_WAVE)
if args.extension or args.low_concurrency:
    accepted.update(
        verify_inference_canaries(
            "results/raw/l40s", "l40s", config_root=CONFIG_ROOT, canary_wave=EXTENSION_CANARY_WAVE
        )
    )
if args.low_concurrency:
    accepted.update(
        verify_inference_canaries(
            "results/raw/l40s", "l40s", config_root=CONFIG_ROOT, canary_wave=LOW_CONCURRENCY_CANARY_WAVE
        )
    )
print(f"Qwen7B layout acceptance passed: {len(accepted)} L40S canaries")
