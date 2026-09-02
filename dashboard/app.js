const state = {
  results: [],
  phaseSummary: [],
  trialSummary: [],
  rankingSummary: [],
  scalingSummary: [],
  precisionSummary: [],
  strategySummary: [],
  communicationSummary: [],
  dataMovementSummary: [],
  memorySummary: [],
  inferenceSummary: [],
  hardwareSummary: [],
  readinessReport: "",
  loading: true,
  sort: { table: "rankingTable", key: "throughput_tokens_sec", direction: -1 },
};

const paths = {
  results: "../results/processed/experiments.csv",
  phaseSummary: "../results/analysis/phase_summary.csv",
  trialSummary: "../results/analysis/trial_summary.csv",
  rankingSummary: "../results/analysis/experiment_rankings.csv",
  scalingSummary: "../results/analysis/scaling_summary.csv",
  precisionSummary: "../results/analysis/precision_summary.csv",
  strategySummary: "../results/analysis/strategy_summary.csv",
  communicationSummary: "../results/analysis/communication_summary.csv",
  dataMovementSummary: "../results/analysis/data_movement_summary.csv",
  memorySummary: "../results/analysis/memory_summary.csv",
  inferenceSummary: "../results/analysis/inference_summary.csv",
  hardwareSummary: "../results/analysis/hardware_summary.csv",
  readinessReport: "../results/processed/readiness_report.md",
};

const analysisFileKeys = {
  "phase_summary.csv": "phaseSummary",
  "trial_summary.csv": "trialSummary",
  "experiment_rankings.csv": "rankingSummary",
  "scaling_summary.csv": "scalingSummary",
  "precision_summary.csv": "precisionSummary",
  "strategy_summary.csv": "strategySummary",
  "communication_summary.csv": "communicationSummary",
  "data_movement_summary.csv": "dataMovementSummary",
  "memory_summary.csv": "memorySummary",
  "inference_summary.csv": "inferenceSummary",
  "hardware_summary.csv": "hardwareSummary",
};

const filterDefinitions = [
  ["platformFilter", "platform_id"],
  ["comparabilityFilter", "comparability"],
  ["phaseFilter", "phase"],
  ["workloadFilter", "workload"],
  ["throughputUnitFilter", "throughput_unit"],
  ["priorityFilter", "priority"],
  ["statusFilter", "status"],
  ["waveFilter", "wave"],
  ["modeFilter", "mode"],
  ["strategyFilter", "strategy"],
  ["precisionFilter", "precision"],
  ["gpuFilter", "gpus"],
];

const rankingColumns = ["experiment_id", "platform_id", "comparability", "workload", "mode", "strategy", "status", "throughput_unit", "throughput_tokens_sec", "runtime_seconds"];
const phaseColumns = ["platform_id", "comparability", "phase", "workload", "throughput_unit", "completed", "failed", "avg_throughput_tokens_sec", "max_memory_used_gb", "energy_joules"];
const trialColumns = ["experiment_id", "platform_id", "comparability", "workload", "throughput_unit", "completed_trials", "failed_trials", "evidence_status", "runtime_mean_seconds", "throughput_tokens_mean", "throughput_tokens_ci95_low", "throughput_tokens_ci95_high"];
const colors = ["#0f766e", "#2563eb", "#7c3aed", "#b45309", "#be123c", "#0891b2", "#4d7c0f"];

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    bindInputs();
    loadDefaultData();
    render();
  });
}

function bindInputs() {
  filterDefinitions.forEach(([id]) => document.getElementById(id).addEventListener("change", render));
  document.getElementById("evidenceFilter").addEventListener("change", render);
  document.getElementById("metricFilter").addEventListener("change", render);
  document.getElementById("resultsFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.results = parseCsv(text);
      rebuildFilters();
      render();
    });
  });
  document.getElementById("analysisFiles").addEventListener("change", async (event) => {
    const files = [...event.target.files];
    const loaded = await Promise.all(files.map(async (file) => ({
      key: analysisFileKeys[file.name],
      rows: parseCsv(await readFile(file)),
    })));
    loaded.forEach(({ key, rows }) => {
      if (key) state[key] = rows;
    });
    rebuildFilters();
    render();
  });
  document.getElementById("readinessFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.readinessReport = text;
      render();
    });
  });
}

async function loadDefaultData() {
  const entries = Object.entries(paths).filter(([key]) => key !== "readinessReport");
  const loaded = await Promise.all(entries.map(async ([key, path]) => [key, await fetchCsv(path)]));
  loaded.forEach(([key, rows]) => {
    state[key] = rows;
  });
  state.readinessReport = await fetchText(paths.readinessReport);
  state.loading = false;
  rebuildFilters();
  render();
}

async function fetchCsv(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    return response.ok ? parseCsv(await response.text()) : [];
  } catch {
    return [];
  }
}

async function fetchText(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    return response.ok ? response.text() : "";
  } catch {
    return "";
  }
}

function readFile(file) {
  return file ? file.text() : Promise.resolve("");
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const [headers, ...dataRows] = rows;
  return headers ? dataRows.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] === undefined ? "" : values[index]]))) : [];
}

function rebuildFilters() {
  filterDefinitions.forEach(([id, key]) => setFilterOptions(id, uniqueValues(state.results, key), key === "gpus"));
  setFilterOptions("evidenceFilter", uniqueValues(state.trialSummary, "evidence_status"));
}

function setFilterOptions(id, values, numeric = false) {
  const select = document.getElementById(id);
  const current = select.value;
  const sorted = numeric ? [...values].sort((left, right) => number(left) - number(right)) : values;
  select.replaceChildren(option("All", ""));
  sorted.forEach((value) => select.append(option(value, value)));
  select.value = sorted.includes(current) ? current : "";
}

function option(label, value) {
  const element = document.createElement("option");
  element.textContent = label;
  element.value = value;
  return element;
}

function uniqueValues(rows, key) {
  return [...new Set(rows.map((row) => rowValue(row, key)).filter(Boolean))].sort((left, right) => String(left).localeCompare(String(right)));
}

function selectedFilters() {
  const filters = Object.fromEntries(filterDefinitions.map(([id, key]) => [key, document.getElementById(id).value]));
  filters.evidence_status = document.getElementById("evidenceFilter").value;
  return filters;
}

function rowValue(row, key) {
  return key === "wave" ? row.submission_bundle_wave || row.environment_submission_bundle_wave || "" : row[key] || "";
}

function matchesFilters(row, filters, supported) {
  return supported.every((key) => !filters[key] || String(rowValue(row, key)) === String(filters[key]));
}

function filteredResults() {
  const filters = selectedFilters();
  return state.results.filter((row) => matchesFilters(row, filters, Object.keys(filters).filter((key) => key !== "evidence_status")));
}

function filteredAnalysisRows(rows, supported) {
  const filters = selectedFilters();
  const unsupported = Object.entries(filters).filter(([, value]) => Boolean(value)).map(([key]) => key).filter((key) => !supported.includes(key));
  if (unsupported.length) {
    return { rows: [], message: "Clear " + unsupported.map(displayName).join(", ") + " to filter this summary safely." };
  }
  return { rows: rows.filter((row) => matchesFilters(row, filters, supported)), message: "" };
}

function render() {
  const rows = filteredResults();
  renderKpis(rows);
  renderReadiness();
  renderProgress(rows);
  renderPhaseBars(rows);
  renderEnergy(rows);
  renderRankingTable();
  renderUncertainty();
  renderStrategy();
  renderPrecision();
  renderMemory();
  renderScaling();
  renderCommunication();
  renderDataMovement();
  renderInference();
  renderPhaseTable();
  renderTrialTable();
  organizeSections();
  document.getElementById("dataStatus").textContent = state.results.length ? rows.length + " of " + state.results.length + " trial rows" : "No data loaded";
}

function renderKpis(rows) {
  const completed = rows.filter((row) => row.status === "completed").length;
  const gpuHours = sum(rows.map((row) => number(row.actual_gpu_hours || row.gpu_hours_estimate)));
  const bestThroughput = max(rows.map(throughput));
  const units = uniqueValues(rows.filter((row) => throughput(row) > 0), "throughput_unit");
  document.getElementById("kpiExperiments").textContent = formatInteger(rows.length);
  document.getElementById("kpiCompleted").textContent = formatInteger(completed);
  document.getElementById("kpiGpuHours").textContent = formatDecimal(gpuHours, 2);
  document.getElementById("kpiBestThroughput").textContent = units.length === 1 ? formatInteger(bestThroughput) : bestThroughput ? "Mixed units" : "—";
}

function renderReadiness() {
  const container = document.getElementById("readinessSummary");
  const status = document.getElementById("readinessStatus");
  const summary = parseReadinessSummary(state.readinessReport);
  container.replaceChildren();
  if (!summary.length) {
    status.textContent = "No report loaded";
    container.append(emptyState("Run python -m hpc_ai_perf.cli refresh, then reload the dashboard."));
    return;
  }
  status.textContent = summary.length + " checks";
  summary.forEach((item) => {
    const card = document.createElement("article");
    card.className = "readiness-item";
    card.innerHTML = "<span>" + escapeHtml(item.label) + "</span><strong>" + escapeHtml(item.value) + "</strong>";
    container.append(card);
  });
}

function parseReadinessSummary(markdown) {
  const lines = markdown.split(/\r?\n/);
  const start = lines.findIndex((line) => line.trim() === "## Summary");
  if (start === -1) return [];
  const items = [];
  for (let index = start + 1; index < lines.length; index += 1) {
    const line = lines[index].trim();
    if (line.startsWith("## ") && index > start + 1) break;
    if (!line.startsWith("- ")) continue;
    const [label, ...rest] = line.slice(2).split(":");
    items.push({ label: label.trim(), value: rest.join(":").trim() });
  }
  return items;
}

function renderProgress(rows) {
  const container = document.getElementById("progressChart");
  const entries = buildCompletionEntries(rows);
  container.replaceChildren();
  if (!entries.length) {
    container.append(emptyState("No recorded trials match the active filters."));
    document.getElementById("progressStatus").textContent = "No trial rows";
    markPanel("progressChart", false, state.results.length > 0, "Trial progress", "experiments.csv");
    return;
  }
  entries.forEach((entry) => {
    const total = sum(Object.values(entry.counts));
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = '<div class="bar-meta"><strong>' + escapeHtml(entry.label) + "</strong><span>" + formatInteger(total) + " trial rows</span></div>";
    const track = document.createElement("div");
    track.className = "status-track";
    ["completed", "failed", "deferred", "other"].forEach((status) => {
      const count = entry.counts[status] || 0;
      if (!count) return;
      const segment = document.createElement("span");
      segment.className = "status-segment status-" + status + "-fill";
      segment.style.width = (count / total) * 100 + "%";
      segment.title = status + ": " + count;
      track.append(segment);
    });
    row.append(track);
    container.append(row);
  });
  const legend = document.createElement("div");
  legend.className = "status-legend";
  [["Completed", "var(--accent)"], ["Failed", "var(--bad)"], ["Deferred", "var(--warn)"], ["Other", "#667085"]].forEach(([label, color]) => {
    const item = document.createElement("span");
    item.textContent = label;
    item.style.setProperty("--legend-color", color);
    legend.append(item);
  });
  container.append(legend);
  document.getElementById("progressStatus").textContent = formatInteger(sum(entries.map((entry) => sum(Object.values(entry.counts))))) + " trial rows";
  markPanel("progressChart", true, true, "Trial progress", "experiments.csv");
}

function buildCompletionEntries(rows) {
  const groups = groupRows(rows, (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified"]);
  return Object.entries(groups).map(([key, group]) => {
    const [platform, comparability] = key.split("\u0000");
    const counts = { completed: 0, failed: 0, deferred: 0, other: 0 };
    group.forEach((row) => {
      const status = ["completed", "failed", "deferred"].includes(row.status) ? row.status : "other";
      counts[status] += 1;
    });
    return { label: platform + " · " + comparability, platform, comparability, counts };
  }).sort((left, right) => left.label.localeCompare(right.label));
}

function renderPhaseBars(rows) {
  const entries = buildPhaseBarEntries(rows);
  renderSeparatedBars("phaseBars", entries, {
    empty: "No throughput values available.",
    value: (entry) => formatInteger(entry.value) + " " + entry.unit,
  });
  markPanel("phaseBars", entries.length > 0, buildPhaseBarEntries(state.results).length > 0, "Phase throughput", "experiments.csv");
}

function buildPhaseBarEntries(rows) {
  const groups = groupRows(rows, (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.phase || "Unassigned", row.workload || "Unassigned", row.throughput_unit || "unit_not_recorded"]);
  return Object.entries(groups).map(([key, group]) => {
    const [platform, comparability, phase, workload, unit] = key.split("\u0000");
    return {
      label: platform + " · " + comparability + " · " + phase + " · " + workload,
      platform, comparability, phase, workload, unit,
      value: average(group.map(throughput).filter((value) => value > 0)),
    };
  }).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value).slice(0, 16);
}

function renderEnergy(rows) {
  const entries = buildEnergyPowerEntries(rows);
  renderSeparatedBars("energyBars", entries, {
    empty: "No measured-region energy evidence is available. Legacy full-window estimates are excluded.",
    barClass: "energy",
    value: (entry) => formatDecimal(entry.value, 0) + " J · " + formatDecimal(entry.power, 1) + " W",
  });
  document.getElementById("energyStatus").textContent = entries.length ? entries.length + " recorded conditions" : "No energy evidence";
  markPanel("energyBars", entries.length > 0, buildEnergyPowerEntries(state.results).length > 0, "Energy and power", "experiments.csv");
}

function buildEnergyPowerEntries(rows) {
  const groups = groupRows(rows.filter((row) => row.energy_scope === "measured_region" && number(row.energy_joules) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.experiment_id || "unknown", row.workload || "unassigned", row.mode || "unassigned"]);
  return Object.entries(groups).map(([key, group]) => {
    const [platform, comparability, experiment, workload, mode] = key.split("\u0000");
    return {
      label: platform + " · " + comparability + " · " + experiment + " · " + workload,
      detail: mode,
      unit: "joules",
      value: average(group.map((row) => number(row.energy_joules))),
      power: average(group.map((row) => number(row.metric_nvidia_smi_power_draw_watts_measured_region || row.metric_power_draw_watts))),
    };
  }).sort((left, right) => right.value - left.value).slice(0, 12);
}

function renderSeparatedBars(containerId, entries, options) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!entries.length) {
    container.append(emptyState(options.empty));
    return;
  }
  const groups = groupRows(entries, (entry) => [entry.unit || "unit_not_recorded"]);
  Object.entries(groups).forEach(([unit, group]) => {
    if (Object.keys(groups).length > 1) {
      const heading = document.createElement("div");
      heading.className = "bar-unit-heading";
      heading.textContent = unit;
      container.append(heading);
    }
    const top = max(group.map((entry) => entry.value));
    group.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = '<div class="bar-meta"><strong>' + escapeHtml(entry.label) + "</strong><span>" + escapeHtml(options.value(entry)) + "</span></div>" + (entry.detail ? '<div class="bar-detail">' + escapeHtml(entry.detail) + "</div>" : "") + '<div class="bar-track"><div class="bar-fill ' + (options.barClass || "") + '" style="width: ' + Math.max(3, (entry.value / top) * 100) + '%"></div></div>';
      container.append(row);
    });
  });
}

function renderRankingTable() {
  const result = filteredAnalysisRows(state.rankingSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "priority", "status", "mode", "strategy", "precision", "gpus"]);
  renderTable("rankingTable", result.rows, rankingColumns, result.message);
  document.getElementById("rankingCount").textContent = result.message || result.rows.length + " rows";
  markPanel("rankingTable", result.rows.length > 0, state.rankingSummary.length > 0, "Experiment rankings", "experiment_rankings.csv");
}

function renderUncertainty() {
  const result = filteredAnalysisRows(state.trialSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "mode", "evidence_status"]);
  const groups = buildUncertaintyGroups(result.rows);
  renderErrorBarCharts("uncertaintyCharts", groups, result.message || "No completed trial throughput evidence.");
  document.getElementById("uncertaintyStatus").textContent = result.message || groups.length + " unit-separated panels";
  markPanel("uncertaintyCharts", groups.length > 0, buildUncertaintyGroups(state.trialSummary).length > 0, "Trial uncertainty", "trial_summary.csv");
}

function buildUncertaintyGroups(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.throughput_tokens_mean) > 0 && number(row.completed_trials) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, unit] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: unit,
      platform, comparability, workload, unit,
      points: group.map((row) => ({
        label: row.experiment_id + " · " + row.workload,
        value: number(row.throughput_tokens_mean),
        low: number(row.throughput_tokens_ci95_low) || number(row.throughput_tokens_mean),
        high: number(row.throughput_tokens_ci95_high) || number(row.throughput_tokens_mean),
      })).sort((left, right) => right.value - left.value).slice(0, 10),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderErrorBarCharts(containerId, groups, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState(emptyMessage));
    return;
  }
  groups.forEach((group) => container.append(errorBarChartCard(group)));
}

function errorBarChartCard(group) {
  const card = chartCard(group.title, group.subtitle);
  const width = 500;
  const height = Math.max(110, group.points.length * 28 + 44);
  const left = 145;
  const right = 25;
  const top = 16;
  const bottom = 22;
  const domain = paddedDomain(group.points.flatMap((point) => [point.low, point.high, point.value]), true);
  const scale = linearScale(domain[0], domain[1], left, width - right);
  const y = (index) => top + index * 28 + 8;
  const ticks = [domain[0], (domain[0] + domain[1]) / 2, domain[1]];
  const grids = ticks.map((value) => '<line class="svg-grid" x1="' + scale(value) + '" y1="' + (top - 8) + '" x2="' + scale(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label" x="' + scale(value) + '" y="' + (height - 5) + '" text-anchor="middle">' + formatCompact(value) + "</text>").join("");
  const marks = group.points.map((point, index) => '<text class="svg-label" x="' + (left - 7) + '" y="' + (y(index) + 3) + '" text-anchor="end">' + escapeSvg(point.label) + '</text><line class="svg-line" stroke="' + colors[index % colors.length] + '" x1="' + scale(point.low) + '" y1="' + y(index) + '" x2="' + scale(point.high) + '" y2="' + y(index) + '"/><circle class="svg-point" fill="' + colors[index % colors.length] + '" cx="' + scale(point.value) + '" cy="' + y(index) + '" r="4"><title>' + escapeSvg(point.label + ": " + formatCompact(point.value) + " (" + formatCompact(point.low) + "–" + formatCompact(point.high) + ")") + "</title></circle>").join("");
  card.append(svgElement(width, height, grids + marks));
  return card;
}

function renderStrategy() {
  const result = filteredAnalysisRows(state.strategySummary, ["platform_id", "comparability", "workload", "throughput_unit", "strategy", "precision", "gpus"]);
  const entries = buildStrategyEntries(result.rows);
  renderSeparatedBars("strategyBars", entries, { empty: result.message || "No completed distributed-strategy evidence.", value: (entry) => formatInteger(entry.value) + " " + entry.unit });
  document.getElementById("strategyStatus").textContent = result.message || entries.length + " conditions";
  markPanel("strategyBars", entries.length > 0, buildStrategyEntries(state.strategySummary).length > 0, "Distributed strategy", "strategy_summary.csv", true);
}

function buildStrategyEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.workload + " · " + row.strategy,
    detail: row.precision + " · " + row.gpus + " GPU" + (number(row.gpus) === 1 ? "" : "s"),
    unit: row.throughput_unit || "unit_not_recorded",
    value: number(row.avg_throughput_tokens_sec),
  })).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value);
}

function renderPrecision() {
  const result = filteredAnalysisRows(state.precisionSummary, ["platform_id", "comparability", "workload", "throughput_unit", "mode", "precision"]);
  const entries = buildPrecisionEntries(result.rows);
  renderSeparatedBars("precisionBars", entries, { empty: result.message || "No completed precision-study evidence.", value: (entry) => formatInteger(entry.value) + " " + entry.unit });
  document.getElementById("precisionStatus").textContent = result.message || entries.length + " conditions";
  markPanel("precisionBars", entries.length > 0, buildPrecisionEntries(state.precisionSummary).length > 0, "Precision trade-off", "precision_summary.csv", true);
}

function buildPrecisionEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.workload + " · " + row.precision,
    detail: row.mode + " · " + formatDecimal(number(row.avg_memory_used_gb), 2) + " GB",
    unit: row.throughput_unit || "unit_not_recorded",
    value: number(row.avg_throughput_tokens_sec),
  })).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value);
}

function renderMemory() {
  const result = filteredAnalysisRows(state.memorySummary, ["platform_id", "comparability", "workload", "throughput_unit"]);
  const series = buildMemorySeries(result.rows);
  renderLineCharts("memoryCharts", series, { empty: result.message || "No completed memory-study evidence. This panel populates after the memory study is collected and analysed.", xLabel: "micro-batch size", yLabel: "max memory (GB)" });
  const capacity = series.filter((item) => item.comparability === "capacity_adjusted");
  renderLineCharts("capacityCharts", capacity, { empty: result.message || "No completed capacity-adjusted L40S evidence.", xLabel: "micro-batch size", yLabel: "max memory (GB)" });
  document.getElementById("memoryStatus").textContent = result.message || series.length + " isolated series";
  document.getElementById("capacityStatus").textContent = result.message || capacity.length + " L40S-only series";
  const allMemorySeries = buildMemorySeries(state.memorySummary);
  markPanel("memoryCharts", series.length > 0, allMemorySeries.length > 0, "Memory optimisation", "memory_summary.csv", true);
  markPanel("capacityCharts", capacity.length > 0, allMemorySeries.some((item) => item.comparability === "capacity_adjusted"), "L40S capacity boundary", "memory_summary.csv", true);
}

function buildMemorySeries(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.micro_batch_size) > 0 && number(row.max_memory_used_gb) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded", row.activation_checkpointing || "unspecified", row.gradient_accumulation_steps || "unspecified"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, unit, checkpointing, accumulation] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: "checkpointing: " + checkpointing + "; accumulation: " + accumulation + "; " + unit,
      platform, comparability, unit,
      points: group.map((row) => ({ x: number(row.micro_batch_size), y: number(row.max_memory_used_gb), label: "batch " + row.micro_batch_size + ": " + formatDecimal(number(row.max_memory_used_gb), 2) + " GB" })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderScaling() {
  const result = filteredAnalysisRows(state.scalingSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "strategy", "precision", "gpus"]);
  const series = buildScalingSeries(result.rows);
  renderLineCharts("scalingCharts", series, { empty: result.message || "No completed scaling-study evidence.", xLabel: "GPU count", yLabel: "speedup" });
  document.getElementById("scalingStatus").textContent = result.message || series.length + " isolated series";
  markPanel("scalingCharts", series.length > 0, buildScalingSeries(state.scalingSummary).length > 0, "Scaling speedup", "scaling_summary.csv", true);
}

function buildScalingSeries(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.gpus) > 0 && number(row.speedup) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.strategy || "unspecified", row.precision || "unspecified", row.throughput_unit || "unit_not_recorded", row.scaling_type || "unspecified"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, strategy, precision, unit, scalingType] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: strategy + "; " + precision + "; " + scalingType + "; " + unit,
      platform, comparability, unit,
      points: group.map((row) => ({ x: number(row.gpus), y: number(row.speedup), label: row.gpus + " GPU: " + formatDecimal(number(row.speedup), 2) + "× speedup; " + formatDecimal(number(row.scaling_efficiency) * 100, 1) + "% efficiency" })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderLineCharts(containerId, series, options) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!series.length) {
    container.append(emptyState(options.empty));
    return;
  }
  series.forEach((item, index) => container.append(lineChartCard(item, options, colors[index % colors.length])));
}

function lineChartCard(series, options, color) {
  const card = chartCard(series.title, series.subtitle);
  const width = 500;
  const height = 220;
  const left = 46;
  const right = 18;
  const top = 15;
  const bottom = 33;
  const xDomain = paddedDomain(series.points.map((point) => point.x), false);
  const yDomain = paddedDomain(series.points.map((point) => point.y), true);
  const x = linearScale(xDomain[0], xDomain[1], left, width - right);
  const y = linearScale(yDomain[0], yDomain[1], height - bottom, top);
  const line = series.points.map((point, index) => (index ? "L" : "M") + x(point.x) + "," + y(point.y)).join(" ");
  const xTicks = [xDomain[0], (xDomain[0] + xDomain[1]) / 2, xDomain[1]];
  const yTicks = [yDomain[0], (yDomain[0] + yDomain[1]) / 2, yDomain[1]];
  const grid = xTicks.map((value) => '<line class="svg-grid" x1="' + x(value) + '" y1="' + top + '" x2="' + x(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label" x="' + x(value) + '" y="' + (height - 18) + '" text-anchor="middle">' + formatCompact(value) + "</text>").join("") + yTicks.map((value) => '<line class="svg-grid" x1="' + left + '" y1="' + y(value) + '" x2="' + (width - right) + '" y2="' + y(value) + '"/><text class="svg-label" x="' + (left - 5) + '" y="' + (y(value) + 3) + '" text-anchor="end">' + formatCompact(value) + "</text>").join("");
  const points = series.points.map((point) => '<circle class="svg-point" fill="' + color + '" cx="' + x(point.x) + '" cy="' + y(point.y) + '" r="4"><title>' + escapeSvg(point.label) + "</title></circle>").join("");
  const labels = '<text class="svg-label" x="' + width / 2 + '" y="' + (height - 3) + '" text-anchor="middle">' + escapeSvg(options.xLabel) + '</text><text class="svg-label" transform="translate(11 ' + height / 2 + ') rotate(-90)" text-anchor="middle">' + escapeSvg(options.yLabel) + "</text>";
  card.append(svgElement(width, height, grid + '<path class="svg-line" stroke="' + color + '" d="' + line + '"/>' + points + labels));
  return card;
}

function renderCommunication() {
  const result = filteredAnalysisRows(state.communicationSummary, ["platform_id", "comparability", "workload", "throughput_unit", "strategy", "gpus"]);
  const entries = buildCommunicationEntries(result.rows);
  renderSeparatedBars("communicationBars", entries, { empty: result.message || "No completed NCCL communication evidence.", barClass: "communication", value: (entry) => formatDecimal(entry.value, 2) + " GB/s" });
  document.getElementById("communicationStatus").textContent = result.message || entries.length + " conditions";
  markPanel("communicationBars", entries.length > 0, buildCommunicationEntries(state.communicationSummary).length > 0, "NCCL communication", "communication_summary.csv", true);
}

function buildCommunicationEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.experiment_id + " · " + row.strategy,
    detail: row.workload + " · " + row.gpus + " GPUs · communication/compute " + formatDecimal(number(row.communication_to_compute_ratio), 2),
    unit: "GB/s",
    value: number(row.avg_nccl_bandwidth_gbps),
  })).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value);
}

function renderDataMovement() {
  const result = filteredAnalysisRows(state.dataMovementSummary, ["platform_id", "comparability", "workload", "throughput_unit"]);
  const entries = buildDataMovementEntries(result.rows);
  renderSeparatedBars("movementBars", entries, { empty: result.message || "No completed data-movement evidence.", value: (entry) => formatDecimal(entry.value, 3) + " s" });
  document.getElementById("movementStatus").textContent = result.message || entries.length + " conditions";
  markPanel("movementBars", entries.length > 0, buildDataMovementEntries(state.dataMovementSummary).length > 0, "Data movement", "data_movement_summary.csv", true);
}

function buildDataMovementEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.experiment_id + " · " + row.workload,
    detail: "workers " + row.num_workers + "; pinned memory " + row.pinned_memory + "; prefetch " + row.prefetch_factor,
    unit: "seconds",
    value: number(row.avg_data_loading_seconds),
  })).filter((entry) => entry.value >= 0).sort((left, right) => right.value - left.value);
}

function renderInference() {
  const result = filteredAnalysisRows(state.inferenceSummary, ["platform_id", "comparability", "workload", "throughput_unit", "precision", "gpus"]);
  const groups = buildInferenceGroups(result.rows);
  renderScatterCharts("inferenceCharts", groups, result.message || "No completed inference latency/throughput evidence.");
  document.getElementById("inferenceStatus").textContent = result.message || groups.length + " unit-separated panels";
  markPanel("inferenceCharts", groups.length > 0, buildInferenceGroups(state.inferenceSummary).length > 0, "Inference latency and throughput", "inference_summary.csv", true);
}

function buildInferenceGroups(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.avg_latency_p95_ms) > 0 && number(row.avg_throughput_tokens_sec) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, unit] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: unit,
      platform, comparability, workload, unit,
      points: group.map((row) => ({ x: number(row.avg_latency_p95_ms), y: number(row.avg_throughput_tokens_sec), label: row.experiment_id + " · " + row.workload + " · " + row.precision + " · " + row.gpus + " GPU" })),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderScatterCharts(containerId, groups, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState(emptyMessage));
    return;
  }
  groups.forEach((group) => container.append(scatterChartCard(group)));
}

function scatterChartCard(group) {
  const card = chartCard(group.title, group.subtitle);
  const width = 500;
  const height = 220;
  const left = 48;
  const right = 18;
  const top = 16;
  const bottom = 33;
  const xDomain = paddedDomain(group.points.map((point) => point.x), false);
  const yDomain = paddedDomain(group.points.map((point) => point.y), true);
  const x = linearScale(xDomain[0], xDomain[1], left, width - right);
  const y = linearScale(yDomain[0], yDomain[1], height - bottom, top);
  const xTicks = [xDomain[0], (xDomain[0] + xDomain[1]) / 2, xDomain[1]];
  const yTicks = [yDomain[0], (yDomain[0] + yDomain[1]) / 2, yDomain[1]];
  const grids = xTicks.map((value) => '<line class="svg-grid" x1="' + x(value) + '" y1="' + top + '" x2="' + x(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label" x="' + x(value) + '" y="' + (height - 18) + '" text-anchor="middle">' + formatCompact(value) + "</text>").join("") + yTicks.map((value) => '<line class="svg-grid" x1="' + left + '" y1="' + y(value) + '" x2="' + (width - right) + '" y2="' + y(value) + '"/><text class="svg-label" x="' + (left - 5) + '" y="' + (y(value) + 3) + '" text-anchor="end">' + formatCompact(value) + "</text>").join("");
  const points = group.points.map((point, index) => '<circle class="svg-point" fill="' + colors[index % colors.length] + '" cx="' + x(point.x) + '" cy="' + y(point.y) + '" r="5"><title>' + escapeSvg(point.label + ": p95 " + formatCompact(point.x) + " ms; " + formatCompact(point.y) + " " + group.unit) + "</title></circle>").join("");
  const labels = '<text class="svg-label" x="' + width / 2 + '" y="' + (height - 3) + '" text-anchor="middle">p95 latency (ms)</text><text class="svg-label" transform="translate(11 ' + height / 2 + ') rotate(-90)" text-anchor="middle">throughput</text>';
  card.append(svgElement(width, height, grids + points + labels));
  return card;
}

function renderPhaseTable() {
  const result = filteredAnalysisRows(state.phaseSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "priority", "mode"]);
  renderTable("phaseTable", result.rows, phaseColumns, result.message);
  document.getElementById("phaseCount").textContent = result.message || result.rows.length + " rows";
  markPanel("phaseTable", result.rows.length > 0, state.phaseSummary.length > 0, "Phase summary", "phase_summary.csv");
}

function renderTrialTable() {
  const result = filteredAnalysisRows(state.trialSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "mode", "evidence_status"]);
  renderTable("trialTable", result.rows, trialColumns, result.message);
  const sufficient = result.rows.filter((row) => row.evidence_status === "sufficient").length;
  document.getElementById("trialCount").textContent = result.message || (result.rows.length ? sufficient + " of " + result.rows.length + " sufficient" : "No trial evidence");
  markPanel("trialTable", result.rows.length > 0, state.trialSummary.length > 0, "Trial evidence", "trial_summary.csv");
}

function markPanel(containerId, visible, sourceAvailable, label, source, includeInPending = false) {
  const panel = document.getElementById(containerId).closest(".panel");
  panel.dataset.hasVisibleData = String(Boolean(visible));
  panel.dataset.sourceAvailable = String(Boolean(sourceAvailable));
  panel.dataset.studyLabel = label;
  panel.dataset.studySource = source;
  panel.dataset.includeInPending = String(includeInPending);
}

function organizeSections() {
  const metricFocus = document.getElementById("metricFilter").value;
  const studySections = ["overview", "performance", "memory", "scaling", "inference", "evidence"];

  document.querySelectorAll(".dashboard-section:not(#pendingStudies) .panel").forEach((panel) => {
    if (!("hasVisibleData" in panel.dataset)) return;
    const metrics = (panel.dataset.metrics || "").split(/\s+/).filter(Boolean);
    const matchesMetric = !metricFocus || metrics.includes(metricFocus);
    panel.hidden = panel.dataset.hasVisibleData !== "true" || !matchesMetric;
  });

  studySections.forEach((sectionId) => {
    const section = document.getElementById(sectionId);
    const hasVisiblePanel = Boolean(section.querySelector(".panel:not([hidden])"));
    section.hidden = !hasVisiblePanel;
    const link = document.querySelector('.section-nav a[href="#' + sectionId + '"]');
    if (link) link.hidden = !hasVisiblePanel;
  });

  const pending = [...document.querySelectorAll('.panel[data-include-in-pending="true"]')]
    .filter((panel) => panel.dataset.sourceAvailable !== "true")
    .map((panel) => ({ label: panel.dataset.studyLabel, source: panel.dataset.studySource }));
  const uniquePending = [...new Map(pending.map((item) => [item.label, item])).values()];
  const pendingSection = document.getElementById("pendingStudies");
  const pendingGrid = document.getElementById("pendingStudyGrid");
  pendingGrid.replaceChildren();
  uniquePending.forEach((item) => {
    const card = document.createElement("article");
    card.className = "pending-card";
    card.innerHTML = "<strong>" + escapeHtml(item.label) + "</strong><span>Awaiting completed rows in " + escapeHtml(item.source) + "</span>";
    pendingGrid.append(card);
  });
  pendingSection.hidden = state.loading || uniquePending.length === 0;
}

function renderTable(id, rows, columns, message = "") {
  const table = document.getElementById(id);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  const sorted = sortRows(rows, id);
  thead.replaceChildren();
  tbody.replaceChildren();
  const header = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = displayName(column);
    th.addEventListener("click", () => setSort(id, column));
    header.append(th);
  });
  thead.append(header);
  if (!sorted.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.className = "empty-state";
    td.colSpan = columns.length;
    td.textContent = message || "No rows available.";
    tr.append(td);
    tbody.append(tr);
    return;
  }
  sorted.slice(0, 120).forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = formatCell(row[column]);
      td.dataset.label = displayName(column);
      if (isNumeric(row[column])) td.classList.add("numeric");
      if (column === "status" && row[column]) td.classList.add("status-" + row[column]);
      tr.append(td);
    });
    tbody.append(tr);
  });
}

function setSort(table, key) {
  if (state.sort.table === table && state.sort.key === key) state.sort.direction *= -1;
  else state.sort = { table, key, direction: 1 };
  render();
}

function sortRows(rows, table) {
  if (state.sort.table !== table) return [...rows];
  return [...rows].sort((left, right) => {
    const leftValue = left[state.sort.key] || "";
    const rightValue = right[state.sort.key] || "";
    const comparison = isNumeric(leftValue) || isNumeric(rightValue) ? number(leftValue) - number(rightValue) : String(leftValue).localeCompare(String(rightValue));
    return comparison * state.sort.direction;
  });
}

function summarizePhases(rows) {
  const grouped = groupRows(rows, (row) => [row.platform_id || "", row.comparability || "", row.phase || "", row.priority || "", row.mode || "", row.workload || "", row.throughput_unit || ""]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform_id, comparability, phase, priority, mode, workload, throughput_unit] = key.split("\u0000");
    return {
      platform_id, comparability, phase, priority, mode, workload, throughput_unit,
      experiments: group.length,
      completed: group.filter((row) => row.status === "completed").length,
      failed: group.filter((row) => row.status === "failed").length,
      avg_throughput_tokens_sec: average(group.map(throughput).filter((value) => value > 0)),
      avg_runtime_seconds: average(group.map((row) => number(row.runtime_seconds)).filter((value) => value > 0)),
      estimated_gpu_hours: sum(group.map((row) => number(row.gpu_hours_estimate))),
    };
  });
}

function groupRows(rows, keyFunction) {
  return rows.reduce((groups, row) => {
    const key = keyFunction(row).join("\u0000");
    groups[key] = groups[key] || [];
    groups[key].push(row);
    return groups;
  }, {});
}

function chartCard(title, subtitle) {
  const card = document.createElement("article");
  card.className = "svg-chart-card";
  const heading = document.createElement("h3");
  heading.textContent = title;
  card.append(heading);
  if (subtitle) {
    const text = document.createElement("p");
    text.textContent = subtitle;
    card.append(text);
  }
  return card;
}

function svgElement(width, height, contents) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  element.setAttribute("class", "svg-chart");
  element.setAttribute("viewBox", "0 0 " + width + " " + height);
  element.setAttribute("role", "img");
  element.setAttribute("aria-label", "Experiment metrics chart");
  element.innerHTML = contents;
  return element;
}

function paddedDomain(values, zeroFloor) {
  const numeric = values.filter((value) => Number.isFinite(value));
  if (!numeric.length) return [0, 1];
  const low = zeroFloor ? 0 : Math.min(...numeric);
  const high = Math.max(...numeric);
  if (high === low) {
    const padding = high ? Math.abs(high) * 0.1 : 1;
    return [Math.max(0, low - padding), high + padding];
  }
  const padding = (high - low) * 0.08;
  return [Math.max(0, low - padding), high + padding];
}

function linearScale(domainStart, domainEnd, rangeStart, rangeEnd) {
  return (value) => rangeStart + ((value - domainStart) / (domainEnd - domainStart || 1)) * (rangeEnd - rangeStart);
}

function throughput(row) {
  return number(row.throughput_tokens_sec || row.metric_throughput_tokens_sec);
}

function number(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function isNumeric(value) {
  return value !== "" && value !== undefined && Number.isFinite(Number.parseFloat(value));
}

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

function max(values) {
  return values.reduce((best, value) => Math.max(best, value || 0), 0);
}

function average(values) {
  const numeric = values.filter((value) => Number.isFinite(value));
  return numeric.length ? sum(numeric) / numeric.length : 0;
}

function displayName(key) {
  return key.replaceAll("_", " ");
}

function formatCell(value) {
  if (!isNumeric(value)) return value || "";
  const numeric = number(value);
  return Math.abs(numeric) >= 1000 ? formatInteger(numeric) : formatDecimal(numeric, 2);
}

function formatInteger(value) {
  return Math.round(value || 0).toLocaleString();
}

function formatDecimal(value, digits) {
  return (value || 0).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function formatCompact(value) {
  return Number(value || 0).toLocaleString(undefined, { notation: "compact", maximumFractionDigits: 2 });
}

function emptyState(message) {
  const element = document.createElement("div");
  element.className = "empty-state";
  element.textContent = message;
  return element;
}

function escapeHtml(value) {
  const element = document.createElement("span");
  element.textContent = value;
  return element.innerHTML;
}

function escapeSvg(value) {
  return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    buildCompletionEntries,
    buildEnergyPowerEntries,
    buildInferenceGroups,
    buildMemorySeries,
    buildPhaseBarEntries,
    buildScalingSeries,
    buildUncertaintyGroups,
    summarizePhases,
  };
}
