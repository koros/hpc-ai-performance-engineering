const assert = require("assert").strict;

const {
  buildCompletionEntries,
  buildEnergyPowerEntries,
  buildExperimentDetail,
  buildInferenceGroups,
  buildMemorySeries,
  buildPhaseBarEntries,
  buildRq1ScalingSeries,
  buildRq2StrategyGroups,
  buildScalingSeries,
  buildUncertaintyGroups,
  comparisonContext,
  summarizePhases,
} = require("./app.js");

const rows = [
  result("l40s", "exact", "gpt2_small", 100),
  result("l40s", "exact", "gpt2_small", 120),
  result("h100", "exact", "gpt2_small", 200),
  result("l40s", "capacity_adjusted", "gpt2_small", 80),
  result("l40s", "exact", "distilbert", 50),
];

run("phase bars never average across platform, comparability, or workload", () => {
  const entries = buildPhaseBarEntries(rows);

  assert.equal(entries.length, 4);
  assert.equal(valueFor(entries, "l40s", "exact", "gpt2_small"), 110);
  assert.equal(valueFor(entries, "h100", "exact", "gpt2_small"), 200);
  assert.equal(valueFor(entries, "l40s", "capacity_adjusted", "gpt2_small"), 80);
  assert.equal(valueFor(entries, "l40s", "exact", "distilbert"), 50);
});

run("fallback phase summaries preserve all statistical grouping fields", () => {
  const summaries = summarizePhases(rows);

  assert.equal(summaries.length, 4);
  assert.equal(summaryFor(summaries, "l40s", "exact", "gpt2_small").avg_throughput_tokens_sec, 110);
  assert.equal(summaryFor(summaries, "h100", "exact", "gpt2_small").avg_throughput_tokens_sec, 200);
  assert.equal(summaryFor(summaries, "l40s", "capacity_adjusted", "gpt2_small").avg_throughput_tokens_sec, 80);
});

run("progress counts retain platform and comparability boundaries", () => {
  const entries = buildCompletionEntries([
    { platform_id: "l40s", comparability: "exact", status: "completed" },
    { platform_id: "h100", comparability: "exact", status: "completed" },
    { platform_id: "l40s", comparability: "capacity_adjusted", status: "failed" },
  ]);

  assert.equal(entries.length, 3);
  assert.equal(entries.find((entry) => entry.platform === "h100").counts.completed, 1);
  assert.equal(entries.find((entry) => entry.comparability === "capacity_adjusted").counts.failed, 1);
});

run("uncertainty panels split platform, comparability, workload, and unit", () => {
  const common = {
    completed_trials: 3,
    throughput_tokens_mean: 100,
    throughput_tokens_ci95_low: 90,
    throughput_tokens_ci95_high: 110,
  };
  const groups = buildUncertaintyGroups([
    { ...common, experiment_id: "E001", platform_id: "l40s", comparability: "exact", workload: "gpt2", throughput_unit: "tokens/s" },
    { ...common, experiment_id: "E002", platform_id: "h100", comparability: "exact", workload: "gpt2", throughput_unit: "tokens/s" },
    { ...common, experiment_id: "E003", platform_id: "l40s", comparability: "capacity_adjusted", workload: "gpt2", throughput_unit: "tokens/s" },
    { ...common, experiment_id: "E004", platform_id: "l40s", comparability: "exact", workload: "distilbert", throughput_unit: "tokens/s" },
    { ...common, experiment_id: "E005", platform_id: "l40s", comparability: "exact", workload: "gpt2", throughput_unit: "samples/s" },
  ]);

  assert.equal(groups.length, 5);
});

run("memory and scaling series never cross platform or comparability boundaries", () => {
  const memoryRows = [
    memory("l40s", "exact", "gpt2", 16, 28),
    memory("l40s", "capacity_adjusted", "gpt2", 25, 43),
    memory("h100", "exact", "gpt2", 32, 51),
  ];
  const scalingRows = [
    scaling("l40s", "exact", "gpt2", 1, 1),
    scaling("l40s", "capacity_adjusted", "gpt2", 2, 1.7),
    scaling("h100", "exact", "gpt2", 2, 1.9),
  ];

  assert.equal(buildMemorySeries(memoryRows).length, 3);
  assert.equal(buildScalingSeries(scalingRows).length, 3);
});

run("research-question charts retain platform and comparability boundaries", () => {
  const scalingGroups = buildRq1ScalingSeries([
    rq1Finding("l40s", "exact", "gpt2", 2, 1.8),
    rq1Finding("h100", "exact", "gpt2", 2, 1.9),
    rq1Finding("l40s", "capacity_adjusted", "gpt2", 2, 1.7),
  ]);
  const strategyGroups = buildRq2StrategyGroups([
    rq2Finding("l40s", "exact", "gpt2", "ddp", 100, 30),
    rq2Finding("h100", "exact", "gpt2", "ddp", 200, 40),
    rq2Finding("l40s", "capacity_adjusted", "gpt2", "ddp", 80, 42),
  ]);

  assert.equal(scalingGroups.length, 3);
  assert.equal(strategyGroups.length, 3);
});

run("energy and inference views retain evidence labels without pooling", () => {
  const energy = buildEnergyPowerEntries([
    energyResult("l40s", "exact", "E008", "gpt2", 1000),
    energyResult("h100", "exact", "E008", "gpt2", 700),
    energyResult("l40s", "capacity_adjusted", "E201", "gpt2", 1200),
  ]);
  const inference = buildInferenceGroups([
    inferenceResult("l40s", "exact", "gpt2"),
    inferenceResult("h100", "exact", "gpt2"),
    inferenceResult("l40s", "exact", "distilbert"),
  ]);

  assert.equal(energy.length, 3);
  assert.equal(inference.length, 3);
});

run("energy views exclude legacy full-window estimates", () => {
  const measured = energyResult("l40s", "exact", "E214", "gpt2", 1000);
  const legacy = {
    ...energyResult("l40s", "exact", "E113", "gpt2", 4000),
    energy_scope: "legacy_full_window",
  };

  const energy = buildEnergyPowerEntries([measured, legacy]);

  assert.equal(energy.length, 1);
  assert.match(energy[0].label, /E214/);
});

run("comparison context reports only genuinely constant controls", () => {
  const context = comparisonContext([
    { platform_id: "l40s", workload: "gpt2", precision: "bf16", gpus: 1 },
    { platform_id: "l40s", workload: "gpt2", precision: "fp16", gpus: 2 },
  ], ["Precision", "GPU count"], ["platform_id", "workload", "precision", "gpus", "missing"]);

  assert.deepEqual(context.changing, ["Precision", "GPU count"]);
  assert.deepEqual(context.constants, [
    { label: "platform id", value: "l40s" },
    { label: "workload", value: "gpt2" },
  ]);
});

run("experiment detail separates completed metrics and omits unavailable groups", () => {
  const detail = buildExperimentDetail([
    {
      condition_id: "E010@l40s", experiment_id: "E010", experiment: "inference_test", platform_id: "l40s",
      comparability: "exact", workload: "gpt2", mode: "inference", status: "completed", repetition_number: 1,
      throughput_tokens_sec: "100", throughput_unit: "generated_tokens_per_second", runtime_seconds: "4",
      metric_nvidia_smi_memory_used_gb_measured_region: "12", environment_observed_gpu_memory_gb: "44.5",
      metric_nvidia_smi_power_draw_watts_measured_region: "180", environment_git_commit: "abc123",
    },
    {
      condition_id: "E010@l40s", experiment_id: "E010", experiment: "inference_test", platform_id: "l40s",
      comparability: "exact", workload: "gpt2", mode: "inference", status: "failed", repetition_number: 2,
      throughput_tokens_sec: "999", runtime_seconds: "1", error: "failed trial",
    },
  ]);

  assert.equal(detail.status, "partial");
  assert.equal(detail.sections.find((section) => section.title === "Performance").items[0].value, 100);
  assert.equal(detail.sections.find((section) => section.title === "Power and energy").items[0].label, "Average power");
  assert.equal(detail.sections.some((section) => section.title === "Distributed communication"), false);
});

function result(platform_id, comparability, workload, throughput_tokens_sec) {
  return {
    platform_id,
    comparability,
    phase: "Baseline profiling",
    priority: "Must Have",
    mode: "training",
    workload,
    throughput_unit: "training_tokens_per_second",
    status: "completed",
    throughput_tokens_sec,
    runtime_seconds: 10,
    gpu_hours_estimate: 1,
  };
}

function memory(platform_id, comparability, workload, micro_batch_size, max_memory_used_gb) {
  return {
    platform_id,
    comparability,
    workload,
    throughput_unit: "tokens/s",
    activation_checkpointing: "false",
    gradient_accumulation_steps: 1,
    micro_batch_size,
    max_memory_used_gb,
  };
}

function scaling(platform_id, comparability, workload, gpus, speedup) {
  return {
    platform_id,
    comparability,
    workload,
    strategy: "ddp",
    precision: "bf16",
    scaling_type: "strong",
    throughput_unit: "tokens/s",
    gpus,
    speedup,
    scaling_efficiency: speedup / gpus,
  };
}

function rq1Finding(platform_id, comparability, workload, gpus, speedup) {
  return {
    platform_id,
    comparability,
    workload,
    precision: "bf16",
    strategy: "ddp",
    throughput_unit: "tokens/s",
    gpus,
    speedup,
    scaling_efficiency: speedup / gpus,
  };
}

function rq2Finding(platform_id, comparability, workload, strategy, throughput_tokens_sec, max_memory_used_gb) {
  return {
    platform_id,
    comparability,
    workload,
    model: "gpt2",
    precision: "bf16",
    strategy,
    throughput_unit: "tokens/s",
    gpus: 2,
    throughput_tokens_sec,
    max_memory_used_gb,
    finding: "Pareto frontier",
  };
}

function energyResult(platform_id, comparability, experiment_id, workload, energy_joules) {
  return {
    platform_id,
    comparability,
    experiment_id,
    workload,
    mode: "training",
    energy_joules,
    energy_scope: "measured_region",
    metric_nvidia_smi_power_draw_watts_measured_region: 200,
  };
}

function inferenceResult(platform_id, comparability, workload) {
  return {
    experiment_id: "E010",
    platform_id,
    comparability,
    workload,
    throughput_unit: "tokens/s",
    avg_latency_p95_ms: 10,
    avg_throughput_tokens_sec: 100,
    precision: "bf16",
    gpus: 1,
  };
}

function valueFor(entries, platform, comparability, workload) {
  return entries.find(
    (entry) =>
      entry.platform === platform && entry.comparability === comparability && entry.workload === workload,
  ).value;
}

function summaryFor(summaries, platform_id, comparability, workload) {
  return summaries.find(
    (summary) =>
      summary.platform_id === platform_id &&
      summary.comparability === comparability &&
      summary.workload === workload,
  );
}

function run(name, testFunction) {
  try {
    testFunction();
    process.stdout.write(`ok - ${name}\n`);
  } catch (error) {
    process.stderr.write(`not ok - ${name}\n${error.stack}\n`);
    process.exitCode = 1;
  }
}
