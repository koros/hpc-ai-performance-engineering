"""Validate model-specific Qwen7B equal-budget acceptance evidence."""

from hpc_ai_perf.config.qwen_layout import CANARY_WAVE, CONFIG_ROOT
from hpc_ai_perf.validation.inference_crossover import verify_inference_canaries

accepted = verify_inference_canaries("results/raw/l40s", "l40s", config_root=CONFIG_ROOT, canary_wave=CANARY_WAVE)
print(f"Qwen7B layout acceptance passed: {len(accepted)} L40S canaries")
