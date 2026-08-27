const assert = require("assert").strict;

const { buildPhaseBarEntries, summarizePhases } = require("./app.js");

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
