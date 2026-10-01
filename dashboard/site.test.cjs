"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const root = __dirname;
const catalog = fs.readFileSync(path.join(root, "index.html"), "utf8");
const findings = fs.readFileSync(path.join(root, "analysis.html"), "utf8");

for (const [name, html, other] of [
  ["catalog", catalog, "analysis.html"],
  ["findings", findings, "index.html"],
]) {
  assert.match(html, new RegExp(`href="${other}"`), `${name} must link to the other page`);
  assert.match(html, /<link rel="icon" type="image\/svg\+xml" href="logo\.svg"\s*\/>/, `${name} must use the dashboard favicon`);
  const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
  assert.ok(scripts.length > 0, `${name} must include its self-contained script`);
  for (const [, script] of scripts) new vm.Script(script, { filename: `${name}.html` });
}
assert.match(fs.readFileSync(path.join(root, "logo.svg"), "utf8"), /<svg\b[^>]*viewBox="0 0 200 200"/);
assert.match(catalog, /Includes verified L40S energy anchors E235–E238, the completed TinyLlama strategy sweep E277–E286, and the TinyLlama weak-scaling wave E287–E293/);
assert.match(findings, /hardware-energy anchors collected on 28 September are listed in the catalog but are not yet included in the findings/);

function embeddedJson(html, variable) {
  const line = html.split("\n").find((entry) => entry.trimStart().startsWith(`var ${variable} = `));
  assert.ok(line, `Missing embedded ${variable}`);
  return JSON.parse(line.slice(line.indexOf("= ") + 2).trim().replace(/;$/, ""));
}

const catalogData = JSON.parse(fs.readFileSync(path.join(root, "experiments-data.json"), "utf8"));
const detailsData = JSON.parse(fs.readFileSync(path.join(root, "experiment-details.json"), "utf8"));
assert.deepStrictEqual(embeddedJson(catalog, "DATA"), catalogData);
assert.deepStrictEqual(embeddedJson(findings, "EXPERIMENT_DETAILS"), detailsData);
assert.equal(Object.keys(catalogData.experiments).length, catalogData.meta.total);
assert.equal(
  catalogData.meta.completed + catalogData.meta.configured + catalogData.meta.deferred,
  catalogData.meta.total,
);
assert.equal(catalogData.meta.completed, 271);
assert.equal(catalogData.meta.configured, 12);
const energyPhase = catalogData.themes.flatMap((theme) => theme.phases)
  .find((phase) => phase.id === "hardware-energy-revalidation");
assert.equal(energyPhase.totals.completed, 4);
assert.equal(energyPhase.totals.configured, 0);
assert.ok(energyPhase.clusters.every((cluster) => cluster.totals.completed === 2));
const strategyPhase = catalogData.themes.flatMap((theme) => theme.phases)
  .find((phase) => phase.id === "distributed-training-strategy");
assert.equal(strategyPhase.totals.completed, 30);
assert.deepStrictEqual(strategyPhase.clusters.slice(-2).map((cluster) => cluster.experiment_ids), [
  ["E277", "E278", "E279", "E280", "E281", "E282", "E283", "E284"],
  ["E285", "E286"],
]);
assert.equal(strategyPhase.clusters.at(-1).constant.gpus, "8");
assert.equal(strategyPhase.clusters.at(-1).constant.per_gpu_batch_size, "2");
assert.equal(strategyPhase.clusters.at(-2).constant.per_gpu_batch_size, "8");
assert.match(findings, /id="nerd-tinyllama-strategy"/);
assert.match(findings, /H100 used a batch of 8 per chip, L40S a batch of 2/);
assert.match(findings, /id="chartTinyllamaStrategySpeed"/);
const weakPhase = catalogData.themes.flatMap((theme) => theme.phases)
  .find((phase) => phase.id === "weak-scaling");
assert.deepStrictEqual(weakPhase.totals, { completed: 15, configured: 0, deferred: 0, total: 15 });
for (const id of ["E287", "E288", "E289"]) {
  assert.equal(catalogData.experiments[id].status, "Completed");
  assert.deepStrictEqual(catalogData.experiments[id].platform_list, ["h100"]);
  assert.equal(detailsData[id].platforms.lovelace_h100.n, 3);
}
assert.match(findings, /id="nerd-tinyllama-weak-scaling"/);
assert.match(findings, /different per-chip batches \(8 on H100, 2 on L40S\)/);
assert.match(findings, /id="chartTinyllamaStrategyMemory"/);
for (let number = 277; number <= 286; number++) {
  const id = `E${number}`;
  const platform = number >= 285 ? "l40s" : "h100";
  const catalogEntry = catalogData.experiments[id];
  const detailsEntry = detailsData[id];
  const paths = fs.readdirSync(path.join(root, "..", "results", "raw", platform, id))
    .filter((name) => name.endsWith(".json"));
  const trials = paths.map((name) => JSON.parse(fs.readFileSync(
    path.join(root, "..", "results", "raw", platform, id, name), "utf8")));
  assert.equal(paths.length, 3);
  assert.ok(trials.every((trial) => trial.status === "completed"));
  assert.equal(catalogEntry.status, "Completed");
  assert.deepStrictEqual(catalogEntry.platform_list, [platform]);
  assert.equal(catalogEntry.platforms[platform].n, 3);
  assert.equal(detailsEntry.platforms[`lovelace_${platform}`].n, 3);
  for (const [field, metric] of [["throughput_tokens_sec", "throughput_tokens_sec"], ["memory_used_gb", "memory_used_gb"]]) {
    const average = trials.reduce((sum, trial) => sum + trial.metrics[metric], 0) / 3;
    assert.ok(Math.abs(catalogEntry.platforms[platform][field] - average) < 0.0001);
  }
}

const matrix = JSON.parse(fs.readFileSync(path.join(root, "../platforms/execution_matrix.json"), "utf8"));
const catalogScript = [...catalog.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)][0][1];
const catalogContext = vm.createContext({});
vm.runInContext(catalogScript.split("\nrenderKpis();")[0], catalogContext);

function expandIds(selectors) {
  return selectors.flatMap((selector) => {
    const match = /^(E)(\d+)-(?:E)?(\d+)$/.exec(selector);
    if (!match) return [selector];
    return Array.from({ length: Number(match[3]) - Number(match[2]) + 1 }, (_, index) =>
      `${match[1]}${String(Number(match[2]) + index).padStart(match[2].length, "0")}`);
  });
}

const matrixOoms = matrix.deferred
  .filter((entry) => entry.observed_failure === "cuda_oom")
  .flatMap((entry) => entry.platforms.flatMap((platform) =>
    expandIds(entry.experiment_ids).map((id) => `${id}@${platform}`)))
  .sort();
const catalogOoms = Object.entries(catalogContext.CAPACITY_HISTORY)
  .flatMap(([id, platforms]) => Object.entries(platforms)
    .filter(([, event]) => event.measuredOom && event.state === "deferred")
    .map(([platform]) => `${id}@${platform}`))
  .sort();
assert.deepStrictEqual(catalogOoms, matrixOoms, "catalog OOM annotations must match deferred matrix policy");
assert.equal(matrixOoms.length, 13);
for (const entry of matrix.deferred.filter((item) => item.observed_failure === "cuda_oom")) {
  assert.equal(entry.status, "deferred");
}
assert.equal(matrix.recovered_failures[0].experiment_id, "E250");
assert.equal(catalogContext.CAPACITY_HISTORY.E250.h100.state, "recovered");
for (const [experimentId, referenceId] of Object.entries({ E235: "E123", E236: "E124", E237: "E127", E238: "E128" })) {
  const anchor = catalogData.experiments[experimentId];
  const trials = fs.readdirSync(path.join(root, "..", "results", "raw", "l40s", experimentId))
    .filter((name) => name.endsWith(".json"))
    .map((name) => JSON.parse(fs.readFileSync(path.join(root, "..", "results", "raw", "l40s", experimentId, name), "utf8")));
  assert.equal(anchor.status, "Completed");
  assert.equal(anchor.energy_reference_experiment_id, referenceId);
  assert.equal(anchor.platforms.l40s.n, 3);
  assert.equal(trials.length, 3);
  assert.ok(trials.every((trial) => trial.status === "completed" && trial.config.metadata.energy_reference_experiment_id === referenceId));
  const meanEnergy = trials.reduce((sum, trial) => sum + trial.metrics.energy_joules_measured_region, 0) / trials.length;
  const meanPower = trials.reduce((sum, trial) => sum + trial.metrics.power_draw_watts, 0) / trials.length;
  assert.ok(Math.abs(anchor.platforms.l40s.energy_joules - meanEnergy) < 0.0001);
  assert.ok(Math.abs(anchor.platforms.l40s.power_draw_watts - meanPower) < 0.0001);
  assert.match(catalogContext.platformStatusHtml(anchor), /state-completed[^>]+role="img"/);
  assert.match(catalogContext.renderExpCard(experimentId), /Measured power \(W\)/);
  assert.match(catalogContext.renderExpCard(experimentId), new RegExp(`Energy reference:[\\s\\S]*${referenceId}`));
}
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E032), "OOM-limited");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E033), "Completed");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E250), "Completed");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E264), "Deferred");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E267), "Deferred");
assert.equal(catalogContext.catalogStatusForPlatform(catalogContext.DATA.experiments.E033, "l40s"), "OOM-limited");
assert.equal(catalogContext.catalogStatusForPlatform(catalogContext.DATA.experiments.E033, "h100"), "Completed");
const mixedPlatformStatus = catalogContext.platformStatusHtml(catalogContext.DATA.experiments.E033);
assert.match(mixedPlatformStatus, /class="platform-marker state-oom-limited" role="img" aria-label="L40S: CUDA out of memory at tested settings"/);
assert.match(mixedPlatformStatus, /class="platform-marker state-completed" role="img" aria-label="H100: completed trial"/);
assert.match(mixedPlatformStatus, /class="platform-dot" aria-hidden="true"><\/span>L40S<\/span>/);
assert.doesNotMatch(mixedPlatformStatus, /status-pill|OOM on/);
assert.match(catalogContext.platformStatusHtml(catalogContext.DATA.experiments.E007), /state-configured[^>]+role="img"[^>]+>.*Configured<\/span>/);
assert.match(catalogContext.platformStatusHtml(catalogContext.DATA.experiments.E089), /state-deferred[^>]+role="img"[^>]+>.*Deferred<\/span>/);
assert.match(catalogContext.platformStatusHtml(catalogContext.DATA.experiments.E250), /state-completed[^>]+role="img"/);
const clusterOf = (id) => catalogContext.DATA.themes.flatMap((theme) => theme.phases)
  .flatMap((phase) => phase.clusters).find((cluster) => cluster.experiment_ids.includes(id));
const firstCluster = catalogContext.renderCluster(catalogContext.DATA.themes[0].phases[0].clusters[0]);
// Result columns carry platform status, so the separate status column only appears without them.
assert.doesNotMatch(firstCluster, /Platform status/);
const strongGpt2 = catalogContext.renderCluster(clusterOf("E033"));
assert.doesNotMatch(strongGpt2, /Platform status/);
assert.match(strongGpt2, /status-pill status-oom-limited" title="L40S: OOM-limited">OOM</);
assert.match(catalogContext.renderCluster(clusterOf("E267")), /status-pill status-deferred" title="H100: Deferred"/);
// Every table shows what changed and the measured outcome, not just IDs and status.
const dataMovement = catalogContext.renderCluster(clusterOf("E097"));
for (const header of ["Pinned memory", "Loader workers", "Prefetch factor", "Tok/s · H100", "Tok/s · L40S"]) {
  assert.match(dataMovement, new RegExp(`<th[^>]*>${header}</th>`), `data movement table must show ${header}`);
}
assert.match(dataMovement, /class="cluster-takeaway"/);
assert.equal(catalogContext.DATA.experiments.E104.prefetch_factor, "8");
assert.match(catalogContext.renderCluster(clusterOf("E154")), /<th>Framework<\/th>[\s\S]*vLLM[\s\S]*TensorRT-LLM/);
assert.match(catalogContext.renderCluster(clusterOf("E281")), /<th[^>]*>Memory GB · H100<\/th>/);
assert.doesNotMatch(firstCluster, /<th>Status<\/th>|<th>Platform\(s\)<\/th>|Platform\(s\):/);
assert.match(catalog, /class="status-legend" aria-label="Platform status key"/);
assert.deepStrictEqual(Array.from(catalogContext.CATALOG_STATUSES), ["Completed", "Configured", "OOM-limited", "Deferred"]);
assert.match(catalog, /red means CUDA OOM at the tested settings, not that a smaller configuration cannot run/);
catalogContext.filters.status = "OOM-limited";
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E032), true);
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E264), false);
catalogContext.filters.status = "Configured";
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E032), false);
assert.ok(Object.values(catalogContext.DATA.experiments).some((e) => catalogContext.matchExp(e)));
catalogContext.filters.platform = "l40s";
catalogContext.filters.status = "OOM-limited";
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E033), true);
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E250), false);
catalogContext.filters.platform = "h100";
catalogContext.filters.status = "Deferred";
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E264), true);
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E032), false);
catalogContext.filters.platform = "h100";
catalogContext.filters.status = "Completed";
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E033), true);
assert.equal(catalogContext.matchExp(catalogContext.DATA.experiments.E032), false);
assert.match(catalogContext.renderExpCard("E033"), /L40S · OOM-limited at tested settings/);
assert.match(catalogContext.renderExpCard("E250"), /H100 · Completed after earlier OOM/);
assert.match(catalogContext.renderExpCard("E264"), /Not run: deferred because/);
assert.doesNotMatch(catalogContext.renderExpCard("E264"), /E264 itself OOMed/);
const shownTotals = catalogContext.catalogTotals(Object.keys(catalogContext.DATA.experiments));
assert.equal(shownTotals.total, catalogData.meta.total);
assert.equal(shownTotals.completed + shownTotals.configured + shownTotals.deferred + shownTotals.oom_limited, shownTotals.total);
assert.equal(shownTotals.oom_limited, 8);
assert.equal(shownTotals.configured, 2);
assert.equal(shownTotals.deferred, 12);

console.log("Dashboard catalog and findings checks passed.");
