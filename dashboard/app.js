const state = {
  results: [],
  phaseSummary: [],
  trialSummary: [],
  readinessReport: "",
  sort: { table: "rankingTable", key: "throughput_tokens_sec", direction: -1 },
};

const paths = {
  results: "../results/processed/experiments.csv",
  phaseSummary: "../results/analysis/phase_summary.csv",
  trialSummary: "../results/analysis/trial_summary.csv",
  readinessReport: "../results/processed/readiness_report.md",
};

const rankingColumns = [
  "condition_id",
  "experiment_id",
  "platform_id",
  "comparability",
  "phase",
  "priority",
  "workload",
  "mode",
  "gpus",
  "precision",
  "strategy",
  "status",
  "throughput_unit",
  "throughput_tokens_sec",
  "runtime_seconds",
  "gpu_hours_estimate",
];

const phaseColumns = [
  "platform_id",
  "comparability",
  "phase",
  "priority",
  "mode",
  "workload",
  "throughput_unit",
  "experiments",
  "completed",
  "failed",
  "avg_throughput_tokens_sec",
  "avg_runtime_seconds",
  "estimated_gpu_hours",
];

const trialColumns = [
  "condition_id",
  "experiment_id",
  "platform_id",
  "comparability",
  "phase",
  "workload",
  "mode",
  "throughput_unit",
  "completed_trials",
  "failed_trials",
  "evidence_status",
  "runtime_mean_seconds",
  "runtime_median_seconds",
  "runtime_ci95_low_seconds",
  "runtime_ci95_high_seconds",
  "throughput_tokens_mean",
  "throughput_tokens_ci95_low",
  "throughput_tokens_ci95_high",
];

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    bindInputs();
    loadDefaultData();
    render();
  });
}

function bindInputs() {
  [
    "platformFilter",
    "comparabilityFilter",
    "phaseFilter",
    "workloadFilter",
    "throughputUnitFilter",
    "priorityFilter",
    "statusFilter",
  ].forEach((id) => {
    document.getElementById(id).addEventListener("change", render);
  });

  document.getElementById("resultsFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.results = parseCsv(text);
      rebuildFilters();
      render();
    });
  });

  document.getElementById("phaseFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.phaseSummary = parseCsv(text);
      render();
    });
  });

  document.getElementById("trialFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.trialSummary = parseCsv(text);
      render();
    });
  });

  document.getElementById("readinessFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.readinessReport = text;
      render();
    });
  });
}

async function loadDefaultData() {
  const [results, phaseSummary, trialSummary, readinessReport] = await Promise.all([
    fetchCsv(paths.results),
    fetchCsv(paths.phaseSummary),
    fetchCsv(paths.trialSummary),
    fetchText(paths.readinessReport),
  ]);
  state.results = results;
  state.phaseSummary = phaseSummary;
  state.trialSummary = trialSummary;
  state.readinessReport = readinessReport;
  rebuildFilters();
  render();
}

async function fetchCsv(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      return [];
    }
    return parseCsv(await response.text());
  } catch {
    return [];
  }
}

async function fetchText(path) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
      return "";
    }
    return response.text();
  } catch {
    return "";
  }
}

function readFile(file) {
  if (!file) {
    return Promise.resolve("");
  }
  return file.text();
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
      if (char === "\r" && next === "\n") {
        index += 1;
      }
      row.push(cell);
      if (row.some((value) => value !== "")) {
        rows.push(row);
      }
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
  if (!headers) {
    return [];
  }

  return dataRows.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] === undefined ? "" : values[index]])),
  );
}

function rebuildFilters() {
  setFilterOptions("platformFilter", uniqueValues(state.results, "platform_id"));
  setFilterOptions("comparabilityFilter", uniqueValues(state.results, "comparability"));
  setFilterOptions("phaseFilter", uniqueValues(state.results, "phase"));
  setFilterOptions("workloadFilter", uniqueValues(state.results, "workload"));
  setFilterOptions("throughputUnitFilter", uniqueValues(state.results, "throughput_unit"));
  setFilterOptions("priorityFilter", uniqueValues(state.results, "priority"));
  setFilterOptions("statusFilter", uniqueValues(state.results, "status"));
}

function setFilterOptions(id, values) {
  const select = document.getElementById(id);
  const current = select.value;
  select.replaceChildren(option("All", ""));
  values.forEach((value) => select.append(option(value, value)));
  select.value = values.includes(current) ? current : "";
}

function option(label, value) {
  const element = document.createElement("option");
  element.textContent = label;
  element.value = value;
  return element;
}

function uniqueValues(rows, key) {
  return [...new Set(rows.map((row) => row[key]).filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function render() {
  const filtered = filteredResults();
  renderKpis(filtered);
  renderBars(filtered);
  renderReadiness();
  renderTable("rankingTable", filtered, rankingColumns);
  renderPhaseTable();
  renderTrialTable();
  document.getElementById("dataStatus").textContent = state.results.length
    ? `${filtered.length} of ${state.results.length} experiments`
    : "No data loaded";
  document.getElementById("rankingCount").textContent = `${filtered.length} rows`;
}

function renderTrialTable() {
  const platform = document.getElementById("platformFilter").value;
  const comparability = document.getElementById("comparabilityFilter").value;
  const phase = document.getElementById("phaseFilter").value;
  const workload = document.getElementById("workloadFilter").value;
  const throughputUnit = document.getElementById("throughputUnitFilter").value;
  const rows = state.trialSummary.filter(
    (row) =>
      (!platform || row.platform_id === platform) &&
      (!comparability || row.comparability === comparability) &&
      (!phase || row.phase === phase) &&
      (!workload || row.workload === workload) &&
      (!throughputUnit || row.throughput_unit === throughputUnit),
  );
  renderTable("trialTable", rows, trialColumns);
  const sufficient = rows.filter((row) => row.evidence_status === "sufficient").length;
  document.getElementById("trialCount").textContent = rows.length
    ? `${sufficient} of ${rows.length} sufficient`
    : "No trial evidence";
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

  status.textContent = `${summary.length} checks`;
  summary.forEach((item) => {
    const card = document.createElement("article");
    card.className = "readiness-item";
    card.innerHTML = `<span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong>`;
    container.append(card);
  });
}

function parseReadinessSummary(markdown) {
  const lines = markdown.split(/\r?\n/);
  const summaryStart = lines.findIndex((line) => line.trim() === "## Summary");
  if (summaryStart === -1) {
    return [];
  }

  const items = [];
  for (let index = summaryStart + 1; index < lines.length; index += 1) {
    const line = lines[index].trim();
    if (line.startsWith("## ") && index > summaryStart + 1) {
      break;
    }
    if (!line.startsWith("- ")) {
      continue;
    }
    const [label, ...rest] = line.slice(2).split(":");
    items.push({ label: label.trim(), value: rest.join(":").trim() });
  }
  return items;
}

function filteredResults() {
  const filters = {
    platform_id: document.getElementById("platformFilter").value,
    comparability: document.getElementById("comparabilityFilter").value,
    phase: document.getElementById("phaseFilter").value,
    workload: document.getElementById("workloadFilter").value,
    throughput_unit: document.getElementById("throughputUnitFilter").value,
    priority: document.getElementById("priorityFilter").value,
    status: document.getElementById("statusFilter").value,
  };

  return state.results.filter((row) =>
    Object.entries(filters).every(([key, value]) => !value || row[key] === value),
  );
}

function renderKpis(rows) {
  const completed = rows.filter((row) => row.status === "completed").length;
  const gpuHours = sum(rows.map((row) => number(row.gpu_hours_estimate)));
  const bestThroughput = max(rows.map((row) => throughput(row)));
  const throughputUnits = uniqueValues(
    rows.filter((row) => throughput(row) > 0),
    "throughput_unit",
  );

  document.getElementById("kpiExperiments").textContent = formatInteger(rows.length);
  document.getElementById("kpiCompleted").textContent = formatInteger(completed);
  document.getElementById("kpiGpuHours").textContent = formatDecimal(gpuHours, 1);
  document.getElementById("kpiBestThroughput").textContent =
    throughputUnits.length > 1 ? "Mixed units" : formatInteger(bestThroughput);
}

function renderBars(rows) {
  const container = document.getElementById("phaseBars");
  const entries = buildPhaseBarEntries(rows);

  container.replaceChildren();
  if (!entries.length) {
    container.append(emptyState("No throughput values available."));
    return;
  }

  const topValue = max(entries.map((entry) => entry.value));
  entries.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-meta"><strong>${escapeHtml(entry.platform)} · ${escapeHtml(entry.comparability)} · ${escapeHtml(entry.phase)} · ${escapeHtml(entry.workload)}</strong><span>${formatInteger(entry.value)} ${escapeHtml(entry.unit)}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width: ${Math.max(3, (entry.value / topValue) * 100)}%"></div></div>
    `;
    container.append(row);
  });
}

function buildPhaseBarEntries(rows) {
  const grouped = rows.reduce((groups, row) => {
    const platform = row.platform_id || "unlabelled";
    const comparability = row.comparability || "unspecified";
    const phase = row.phase || "Unassigned";
    const workload = row.workload || "Unassigned";
    const unit = row.throughput_unit || "unit_not_recorded";
    const key = `${platform}\u0000${comparability}\u0000${phase}\u0000${workload}\u0000${unit}`;
    groups[key] = groups[key] || [];
    groups[key].push(row);
    return groups;
  }, {});
  const entries = Object.entries(grouped)
    .map(([key, groupRows]) => ({
      platform: key.split("\u0000")[0],
      comparability: key.split("\u0000")[1],
      phase: key.split("\u0000")[2],
      workload: key.split("\u0000")[3],
      unit: key.split("\u0000")[4],
      value: average(groupRows.map((row) => throughput(row))),
    }))
    .filter((entry) => entry.value > 0)
    .sort((left, right) => right.value - left.value)
    .slice(0, 10);
  return entries;
}

function renderPhaseTable() {
  const filters = {
    platform_id: document.getElementById("platformFilter").value,
    comparability: document.getElementById("comparabilityFilter").value,
    phase: document.getElementById("phaseFilter").value,
    workload: document.getElementById("workloadFilter").value,
    throughput_unit: document.getElementById("throughputUnitFilter").value,
    priority: document.getElementById("priorityFilter").value,
  };
  const hasActiveFilter = Object.values(filters).some(Boolean) || document.getElementById("statusFilter").value;
  const rows = state.phaseSummary.length && !hasActiveFilter ? state.phaseSummary : summarizePhases(filteredResults());
  renderTable("phaseTable", rows, phaseColumns);
  document.getElementById("phaseCount").textContent = `${rows.length} rows`;
}

function renderTable(id, rows, columns) {
  const table = document.getElementById(id);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  const sortedRows = sortRows(rows, id);

  thead.replaceChildren();
  tbody.replaceChildren();

  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column.replaceAll("_", " ");
    th.addEventListener("click", () => setSort(id, column));
    headerRow.append(th);
  });
  thead.append(headerRow);

  if (!sortedRows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.className = "empty-state";
    td.colSpan = columns.length;
    td.textContent = "No rows available.";
    tr.append(td);
    tbody.append(tr);
    return;
  }

  sortedRows.slice(0, 120).forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = formatCell(row[column]);
      if (isNumeric(row[column])) {
        td.classList.add("numeric");
      }
      if (column === "status" && row[column]) {
        td.classList.add(`status-${row[column]}`);
      }
      tr.append(td);
    });
    tbody.append(tr);
  });
}

function setSort(table, key) {
  if (state.sort.table === table && state.sort.key === key) {
    state.sort.direction *= -1;
  } else {
    state.sort = { table, key, direction: 1 };
  }
  render();
}

function sortRows(rows, table) {
  if (state.sort.table !== table) {
    return [...rows];
  }

  return [...rows].sort((left, right) => {
    const leftValue = left[state.sort.key] || "";
    const rightValue = right[state.sort.key] || "";
    if (isNumeric(leftValue) || isNumeric(rightValue)) {
      return (number(leftValue) - number(rightValue)) * state.sort.direction;
    }
    return leftValue.localeCompare(rightValue) * state.sort.direction;
  });
}

function summarizePhases(rows) {
  const grouped = rows.reduce((groups, row) => {
    const key = [
      row.platform_id || "",
      row.comparability || "",
      row.phase || "",
      row.priority || "",
      row.mode || "",
      row.workload || "",
      row.throughput_unit || "",
    ].join("\u0000");
    groups[key] = groups[key] || [];
    groups[key].push(row);
    return groups;
  }, {});
  return Object.entries(grouped).map(([key, groupRows]) => ({
    platform_id: key.split("\u0000")[0],
    comparability: key.split("\u0000")[1],
    phase: key.split("\u0000")[2],
    priority: key.split("\u0000")[3],
    mode: key.split("\u0000")[4],
    workload: key.split("\u0000")[5],
    throughput_unit: key.split("\u0000")[6],
    experiments: groupRows.length,
    completed: groupRows.filter((row) => row.status === "completed").length,
    failed: groupRows.filter((row) => row.status === "failed").length,
    avg_throughput_tokens_sec: average(groupRows.map((row) => throughput(row))),
    avg_runtime_seconds: average(groupRows.map((row) => number(row.runtime_seconds))),
    estimated_gpu_hours: sum(groupRows.map((row) => number(row.gpu_hours_estimate))),
  }));
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
  const numeric = values.filter((value) => Number.isFinite(value) && value > 0);
  return numeric.length ? sum(numeric) / numeric.length : 0;
}

function formatCell(value) {
  if (!isNumeric(value)) {
    return value || "";
  }
  const numeric = number(value);
  if (Math.abs(numeric) >= 1000) {
    return formatInteger(numeric);
  }
  return formatDecimal(numeric, 2);
}

function formatInteger(value) {
  return Math.round(value || 0).toLocaleString();
}

function formatDecimal(value, digits) {
  return (value || 0).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
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

if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildPhaseBarEntries, summarizePhases };
}
