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

console.log("Dashboard catalog and findings checks passed.");
