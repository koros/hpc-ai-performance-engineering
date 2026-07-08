const state = {
  results: [],
  phaseSummary: [],
  sort: { table: "rankingTable", key: "throughput_tokens_sec", direction: -1 },
};

const paths = {
  results: "../results/processed/experiments.csv",
  phaseSummary: "../results/analysis/phase_summary.csv",
};

const rankingColumns = [
  "experiment_id",
  "phase",
  "priority",
  "workload",
  "mode",
  "gpus",
  "precision",
  "strategy",
  "status",
  "throughput_tokens_sec",
  "runtime_seconds",
  "gpu_hours_estimate",
];

const phaseColumns = [
  "phase",
  "priority",
  "mode",
  "workload",
  "experiments",
  "completed",
  "failed",
  "avg_throughput_tokens_sec",
  "avg_runtime_seconds",
  "estimated_gpu_hours",
];

document.addEventListener("DOMContentLoaded", () => {
  bindInputs();
  loadDefaultData();
  render();
});

function bindInputs() {
  ["phaseFilter", "workloadFilter", "priorityFilter", "statusFilter"].forEach((id) => {
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
}

async function loadDefaultData() {
  const [results, phaseSummary] = await Promise.all([fetchCsv(paths.results), fetchCsv(paths.phaseSummary)]);
  state.results = results;
  state.phaseSummary = phaseSummary;
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
  setFilterOptions("phaseFilter", uniqueValues(state.results, "phase"));
  setFilterOptions("workloadFilter", uniqueValues(state.results, "workload"));
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
  renderTable("rankingTable", filtered, rankingColumns);
  renderPhaseTable();
  document.getElementById("dataStatus").textContent = state.results.length
    ? `${filtered.length} of ${state.results.length} experiments`
    : "No data loaded";
  document.getElementById("rankingCount").textContent = `${filtered.length} rows`;
}

function filteredResults() {
  const filters = {
    phase: document.getElementById("phaseFilter").value,
    workload: document.getElementById("workloadFilter").value,
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

  document.getElementById("kpiExperiments").textContent = formatInteger(rows.length);
  document.getElementById("kpiCompleted").textContent = formatInteger(completed);
  document.getElementById("kpiGpuHours").textContent = formatDecimal(gpuHours, 1);
  document.getElementById("kpiBestThroughput").textContent = formatInteger(bestThroughput);
}

function renderBars(rows) {
  const container = document.getElementById("phaseBars");
  const grouped = groupBy(rows, "phase");
  const entries = Object.entries(grouped)
    .map(([phase, groupRows]) => ({
      phase: phase || "Unassigned",
      value: average(groupRows.map((row) => throughput(row))),
    }))
    .filter((entry) => entry.value > 0)
    .sort((left, right) => right.value - left.value)
    .slice(0, 10);

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
      <div class="bar-meta"><strong>${escapeHtml(entry.phase)}</strong><span>${formatInteger(entry.value)} tokens/s</span></div>
      <div class="bar-track"><div class="bar-fill" style="width: ${Math.max(3, (entry.value / topValue) * 100)}%"></div></div>
    `;
    container.append(row);
  });
}

function renderPhaseTable() {
  const rows = state.phaseSummary.length ? state.phaseSummary : summarizePhases(filteredResults());
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
  return Object.entries(groupBy(rows, "phase")).map(([phase, groupRows]) => ({
    phase,
    priority: "",
    mode: "",
    workload: "",
    experiments: groupRows.length,
    completed: groupRows.filter((row) => row.status === "completed").length,
    failed: groupRows.filter((row) => row.status === "failed").length,
    avg_throughput_tokens_sec: average(groupRows.map((row) => throughput(row))),
    avg_runtime_seconds: average(groupRows.map((row) => number(row.runtime_seconds))),
    estimated_gpu_hours: sum(groupRows.map((row) => number(row.gpu_hours_estimate))),
  }));
}

function groupBy(rows, key) {
  return rows.reduce((groups, row) => {
    const value = row[key] || "";
    groups[value] = groups[value] || [];
    groups[value].push(row);
    return groups;
  }, {});
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
