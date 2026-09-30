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
  assert.match(html, /L40S energy revalidation trials from 28 September are not included/);
  const scripts = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
  assert.ok(scripts.length > 0, `${name} must include its self-contained script`);
  for (const [, script] of scripts) new vm.Script(script, { filename: `${name}.html` });
}

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
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E032), "OOM-limited");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E033), "Completed");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E250), "Completed");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E264), "Deferred");
assert.equal(catalogContext.catalogStatus(catalogContext.DATA.experiments.E267), "Deferred");
assert.equal(catalogContext.catalogStatusForPlatform(catalogContext.DATA.experiments.E033, "l40s"), "OOM-limited");
assert.equal(catalogContext.catalogStatusForPlatform(catalogContext.DATA.experiments.E033, "h100"), "Completed");
assert.match(catalogContext.platformStatusHtml(catalogContext.DATA.experiments.E033), /L40S · <span class="status-pill status-oom-limited">OOM-limited<\/span>/);
assert.match(catalogContext.platformStatusHtml(catalogContext.DATA.experiments.E033), /H100 · completed/);
assert.deepStrictEqual(Array.from(catalogContext.CATALOG_STATUSES), ["Completed", "Configured", "OOM-limited", "Deferred"]);
assert.match(catalog, /<code>OOM-limited<\/code> means a CUDA out-of-memory failure was measured/);
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
assert.equal(shownTotals.configured, 6);
assert.equal(shownTotals.deferred, 12);

console.log("Dashboard catalog and findings checks passed.");
