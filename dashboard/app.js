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
  researchQuestionSummary: [],
  researchFindingSummary: [],
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
  researchQuestionSummary: "../results/analysis/research_question_summary.csv",
  researchFindingSummary: "../results/analysis/research_finding_summary.csv",
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
  "research_question_summary.csv": "researchQuestionSummary",
  "research_finding_summary.csv": "researchFindingSummary",
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
  ["researchQuestionFilter", "research_question"],
];

const rankingColumns = ["experiment_id", "platform_id", "comparability", "workload", "status", "throughput_tokens_sec", "throughput_unit", "runtime_seconds"];
const phaseColumns = ["platform_id", "comparability", "phase", "workload", "completed", "failed", "avg_throughput_tokens_sec", "max_memory_used_gb"];
const trialColumns = ["experiment_id", "platform_id", "comparability", "workload", "completed_trials", "failed_trials", "evidence_status", "throughput_tokens_mean", "throughput_tokens_ci95_low", "throughput_tokens_ci95_high"];
const colors = ["#0f766e", "#2563eb", "#7c3aed", "#b45309", "#be123c", "#0891b2", "#4d7c0f"];

const detailMetricSections = [
  ["Performance", [
    ["Throughput", ["throughput_tokens_sec", "metric_throughput_tokens_sec"], "throughput_unit", 1],
    ["Sample throughput", ["throughput_samples_sec", "metric_throughput_samples_sec"], "samples/s", 2],
    ["Runtime", ["runtime_seconds"], "s", 2],
    ["Mean step time", ["metric_step_time_mean_ms"], "ms", 2],
    ["p95 step time", ["metric_step_time_p95_ms"], "ms", 2],
  ]],
  ["Memory", [
    ["GPU memory used", ["metric_nvidia_smi_memory_used_gb_measured_region", "metric_memory_used_gb", "memory_used_gb"], "GB", 2],
    ["Torch memory allocated", ["metric_torch_memory_memory_used_gb"], "GB", 2],
    ["Torch memory reserved", ["metric_torch_memory_memory_reserved_gb", "metric_memory_reserved_gb"], "GB", 2],
  ]],
  ["GPU telemetry", [
    ["Average GPU utilisation", ["metric_nvidia_smi_gpu_utilization_measured_region", "metric_gpu_utilization", "gpu_utilization"], "%", 1],
    ["Peak GPU utilisation", ["metric_nvidia_smi_gpu_utilization_max_measured_region", "metric_nvidia_smi_gpu_utilization_max"], "%", 1],
    ["Average SM clock", ["metric_nvidia_smi_sm_clock_mean_mhz_measured_region", "metric_nvidia_smi_sm_clock_mean_mhz"], "MHz", 0],
    ["Peak temperature", ["metric_nvidia_smi_temperature_max_c_measured_region", "metric_nvidia_smi_temperature_max_c"], "°C", 1],
  ]],
  ["Power and energy", [
    ["Measured-region energy", ["metric_nvidia_smi_energy_joules_measured_region", "energy_joules"], "J", 1],
    ["Average power", ["metric_nvidia_smi_power_draw_watts_measured_region", "metric_power_draw_watts", "power_draw_watts"], "W", 1],
    ["Peak power", ["metric_nvidia_smi_power_draw_max_watts_measured_region", "metric_power_draw_max_watts"], "W", 1],
    ["Measurement duration", ["metric_nvidia_smi_measurement_duration_seconds", "measurement_duration_seconds"], "s", 2],
  ]],
  ["CPU and data movement", [
    ["Torch profiler CPU time", ["metric_torch_profiler_self_cpu_time_ms"], "ms", 2],
    ["Data loading", ["metric_data_loading_seconds"], "s", 3],
    ["CPU-to-GPU transfer", ["metric_cpu_to_gpu_transfer_seconds"], "s", 3],
  ]],
  ["Inference", [
    ["p50 latency", ["metric_latency_p50_ms", "latency_p50_ms"], "ms", 2],
    ["p95 latency", ["metric_latency_p95_ms", "latency_p95_ms"], "ms", 2],
    ["p50 time to first token", ["metric_time_to_first_token_p50_ms"], "ms", 2],
    ["p95 time to first token", ["metric_time_to_first_token_p95_ms"], "ms", 2],
    ["Generated tokens", ["metric_generated_tokens"], "tokens", 0],
    ["Requests per second", ["metric_throughput_requests_sec"], "requests/s", 2],
  ]],
  ["Distributed communication", [
    ["NCCL bandwidth", ["metric_nccl_bandwidth_gbps"], "GB/s", 2],
    ["All-reduce time", ["metric_allreduce_time_seconds"], "s", 4],
    ["CUDA event time", ["metric_cuda_event_seconds"], "s", 4],
  ]],
];

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
  document.getElementById("resetFilters").addEventListener("click", resetFilters);
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
  document.getElementById("experimentDialogClose").addEventListener("click", closeExperimentDialog);
  document.getElementById("experimentDialog").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeExperimentDialog();
  });
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-condition-ids]");
    if (target) openExperimentDialog(target.dataset.conditionIds.split(","));
  });
  document.addEventListener("keydown", (event) => {
    const target = event.target.closest("[data-condition-ids]");
    if (target && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      openExperimentDialog(target.dataset.conditionIds.split(","));
    }
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
  setFilterOptions("researchQuestionFilter", uniqueValues(state.researchQuestionSummary, "research_question"));
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
  if (key === "research_question") return row.research_question || "";
  return key === "wave" ? row.submission_bundle_wave || row.environment_submission_bundle_wave || "" : row[key] || "";
}

function matchesFilters(row, filters, supported) {
  return supported.every((key) => {
    if (!filters[key]) return true;
    if (key === "research_question") return researchQuestionsForRow(row).includes(filters[key]);
    return String(rowValue(row, key)) === String(filters[key]);
  });
}

function researchQuestionsForRow(row) {
  if (row.research_question) return [row.research_question];
  try {
    const value = JSON.parse(row.research_questions || "[]");
    return Array.isArray(value) ? value : [];
  } catch {
    return String(row.research_questions || "").match(/RQ[1-7]/g) || [];
  }
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
  renderActiveFilters();
  renderResearchQuestions();
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

function renderResearchQuestions() {
  const coverage = filteredAnalysisRows(state.researchQuestionSummary, ["platform_id", "comparability", "research_question"]);
  renderResearchCoverage(coverage.rows, coverage.message);
  document.getElementById("researchCoverageStatus").textContent = coverage.message || researchCoverageLabel(coverage.rows);
  markPanel("researchQuestionCards", coverage.rows.length > 0, state.researchQuestionSummary.length > 0, "Research-question coverage", "research_question_summary.csv");

  const findings = filteredAnalysisRows(state.researchFindingSummary, ["platform_id", "comparability", "workload", "precision", "strategy", "gpus", "research_question"]);
  const rq1 = findings.rows.filter((row) => row.research_question === "RQ1");
  const rq2 = findings.rows.filter((row) => row.research_question === "RQ2");
  const rq1Series = buildRq1ScalingSeries(rq1);
  const rq2Groups = buildRq2StrategyGroups(rq2);
  renderLineCharts("rq1ScalingCharts", rq1Series, {
    empty: findings.message || "No completed RQ1 scaling finding is available for the active filters.",
    xLabel: "GPU count",
    yLabel: "speedup",
  });
  renderStrategyTradeoffCharts("rq2StrategyCharts", rq2Groups, findings.message || "No matched RQ2 strategy findings are available for the active filters.");
  document.getElementById("rq1Status").textContent = findings.message || (rq1Series.length ? rq1Series.length + " isolated scaling series" : "No scaling findings");
  document.getElementById("rq2Status").textContent = findings.message || (rq2Groups.length ? rq2Groups.length + " matched strategy groups" : "No matched strategy findings");
  markPanel("rq1ScalingCharts", rq1Series.length > 0, state.researchFindingSummary.some((row) => row.research_question === "RQ1"), "RQ1 scaling efficiency", "research_finding_summary.csv", true);
  markPanel("rq2StrategyCharts", rq2Groups.length > 0, state.researchFindingSummary.some((row) => row.research_question === "RQ2"), "RQ2 strategy trade-off", "research_finding_summary.csv", true);
  renderResearchFindingList(findings.rows, findings.message);
  document.getElementById("researchFindingStatus").textContent = findings.message || (findings.rows.length ? findings.rows.length + " traceable finding" + (findings.rows.length === 1 ? "" : "s") : "No findings");
  markPanel("researchFindingList", findings.rows.length > 0, state.researchFindingSummary.length > 0, "Traceable research findings", "research_finding_summary.csv", true);
}

function renderResearchCoverage(rows, message) {
  const container = document.getElementById("researchQuestionCards");
  container.replaceChildren();
  if (!rows.length) {
    container.append(emptyState(message || "No research-question coverage is available. Run the refresh pipeline after collecting accepted evidence."));
    return;
  }
  rows.sort((left, right) => (left.research_question + left.platform_id + left.comparability).localeCompare(right.research_question + right.platform_id + right.comparability)).forEach((row) => {
    const card = document.createElement("article");
    const awaiting = row.evidence_status.startsWith("awaiting") || row.evidence_status === "insufficient_data";
    card.className = "research-question-card" + (awaiting ? " is-pending" : "");
    const conditionLabel = number(row.direct_conditions) === 1 ? "condition" : "conditions";
    card.innerHTML = '<div class="research-card-heading"><strong>' + escapeHtml(row.research_question) + " · " + escapeHtml(row.objective) + '</strong><span class="evidence-badge evidence-' + escapeHtml(row.evidence_status) + '">' + escapeHtml(displayEvidenceStatus(row.evidence_status)) + "</span></div>" +
      '<p class="research-question-text">' + escapeHtml(row.question) + "</p>" +
      '<dl class="research-metrics"><div><dt>Scope</dt><dd>' + escapeHtml((row.platform_id || "not recorded") + (row.comparability ? " · " + row.comparability : "")) + '</dd></div><div><dt>Direct evidence</dt><dd>' + formatInteger(number(row.direct_conditions)) + " " + conditionLabel + " · " + formatInteger(number(row.completed_trials)) + "/" + formatInteger(number(row.required_trials)) + " trials</dd></div></dl>" +
      '<p class="research-note">' + escapeHtml(row.scope_note || row.limitation) + "</p>";
    container.append(card);
  });
}

function researchCoverageLabel(rows) {
  const sufficient = rows.filter((row) => row.evidence_status === "sufficient" || row.evidence_status === "supporting_platform_evidence").length;
  return sufficient + " coverage entries with completed evidence";
}

function buildRq1ScalingSeries(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.speedup) > 0 && number(row.gpus) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.precision || "unspecified", row.strategy || "unspecified", row.throughput_unit || "unit_not_recorded"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, precision, strategy, unit] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: precision + "; " + strategy + "; " + unit,
      platform, comparability, workload, unit,
      comparison: comparisonContext(group, ["GPU count"], ["platform_id", "comparability", "workload", "precision", "strategy", "throughput_unit"]),
      points: group.map((row) => ({
        x: number(row.gpus),
        y: number(row.speedup),
        label: row.gpus + " GPUs: " + formatDecimal(number(row.speedup), 2) + "× speedup; " + formatDecimal(number(row.scaling_efficiency) * 100, 1) + "% efficiency; " + row.source_condition_ids,
        conditionIds: parseConditionIds(row.source_condition_ids),
      })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function buildRq2StrategyGroups(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.throughput_tokens_sec) > 0 && number(row.max_memory_used_gb) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.model || "unassigned", row.precision || "unspecified", row.gpus || "unspecified", row.throughput_unit || "unit_not_recorded"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, model, precision, gpus, unit] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: model + "; " + precision + "; " + gpus + " GPUs; " + unit,
      platform, comparability, workload, unit,
      comparison: comparisonContext(group, ["Distributed strategy"], ["platform_id", "comparability", "workload", "model", "precision", "gpus", "throughput_unit"]),
      points: group.map((row) => ({
        x: number(row.max_memory_used_gb),
        y: number(row.throughput_tokens_sec),
        label: row.strategy + " · " + row.finding,
        status: row.finding.includes("Pareto frontier") ? "Pareto frontier" : "Dominated",
        conditionIds: parseConditionIds(row.source_condition_ids),
      })).sort((left, right) => right.y - left.y),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderStrategyTradeoffCharts(containerId, groups, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState(emptyMessage));
    return;
  }
  groups.forEach((group) => container.append(strategyTradeoffChartCard(group)));
}

function strategyTradeoffChartCard(group) {
  const card = chartCard(group.title, group.subtitle, group.comparison);
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
  const points = group.points.map((point) => {
    const color = point.status === "Pareto frontier" ? "#0f766e" : "#94a3b8";
    return '<circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="' + color + '" cx="' + x(point.x) + '" cy="' + y(point.y) + '" r="5"><title>' + escapeSvg(point.label + "; " + formatCompact(point.x) + " GB; " + formatCompact(point.y) + " " + group.unit) + "</title></circle>";
  }).join("");
  const labels = '<text class="svg-label" x="' + width / 2 + '" y="' + (height - 3) + '" text-anchor="middle">maximum GPU memory (GB)</text><text class="svg-label" transform="translate(11 ' + height / 2 + ') rotate(-90)" text-anchor="middle">throughput</text>';
  card.append(svgElement(width, height, grids + points + labels));
  return card;
}

function renderResearchFindingList(rows, message) {
  const container = document.getElementById("researchFindingList");
  container.replaceChildren();
  if (!rows.length) {
    container.append(emptyState(message || "No traceable RQ1 or RQ2 finding is available for the active filters."));
    return;
  }
  rows.sort((left, right) => (left.research_question + left.platform_id + left.workload + left.strategy).localeCompare(right.research_question + right.platform_id + right.workload + right.strategy)).forEach((row) => {
    const item = document.createElement("article");
    item.className = "research-finding";
    setConditionTarget(item, "", parseConditionIds(row.source_condition_ids));
    item.innerHTML = '<div class="research-card-heading"><strong>' + escapeHtml(row.research_question + " · " + row.finding_type.replaceAll("_", " ")) + '</strong><span class="evidence-badge evidence-' + escapeHtml(row.finding_status) + '">' + escapeHtml(displayEvidenceStatus(row.finding_status)) + "</span></div>" +
      '<p>' + escapeHtml(row.finding) + "</p>" +
      '<dl class="research-metrics"><div><dt>Scope</dt><dd>' + escapeHtml([row.platform_id, row.comparability, row.workload, row.precision, row.strategy].filter(Boolean).join(" · ")) + '</dd></div><div><dt>Trial evidence</dt><dd>' + formatInteger(number(row.completed_trials)) + "/" + formatInteger(number(row.required_trials)) + " trials</dd></div><div><dt>Source conditions</dt><dd>" + escapeHtml(row.source_condition_ids) + "</dd></div></dl>" +
      '<p class="research-note">' + escapeHtml(row.limitation) + "</p>";
    container.append(item);
  });
}

function displayEvidenceStatus(value) {
  return String(value || "not available").replaceAll("_", " ");
}

function renderKpis(rows) {
  const completed = rows.filter((row) => row.status === "completed").length;
  const conditions = new Set(rows.map(conditionId).filter(Boolean)).size;
  const gpuHours = sum(rows.map((row) => number(row.actual_gpu_hours || row.gpu_hours_estimate)));
  const bestThroughput = max(rows.map(throughput));
  const units = uniqueValues(rows.filter((row) => throughput(row) > 0), "throughput_unit");
  document.getElementById("kpiExperiments").textContent = formatInteger(conditions);
  document.getElementById("kpiCompleted").textContent = formatInteger(completed);
  document.getElementById("kpiGpuHours").textContent = formatDecimal(gpuHours, 2);
  document.getElementById("kpiBestThroughput").textContent = units.length === 1 ? formatInteger(bestThroughput) : bestThroughput ? "Mixed units" : "—";
}

function renderActiveFilters() {
  const container = document.getElementById("activeFilters");
  const filters = selectedFilters();
  const metric = document.getElementById("metricFilter").value;
  const active = Object.entries(filters).filter(([, value]) => Boolean(value)).map(([key, value]) => ({ key, label: displayName(key), value }));
  if (metric) active.push({ key: "metric", label: "metric focus", value: metric });
  container.replaceChildren();
  if (!active.length) {
    const empty = document.createElement("span");
    empty.className = "filter-empty";
    empty.textContent = "All recorded evidence";
    container.append(empty);
    return;
  }
  active.forEach((item) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "active-filter";
    chip.innerHTML = "<span>" + escapeHtml(item.label) + "</span>" + escapeHtml(item.value) + " <b aria-hidden=\"true\">×</b>";
    chip.setAttribute("aria-label", "Clear " + item.label + " filter");
    chip.addEventListener("click", () => clearFilter(item.key));
    container.append(chip);
  });
}

function clearFilter(key) {
  if (key === "metric") document.getElementById("metricFilter").value = "";
  else {
    const definition = filterDefinitions.find(([, filterKey]) => filterKey === key);
    if (definition) document.getElementById(definition[0]).value = "";
    if (key === "evidence_status") document.getElementById("evidenceFilter").value = "";
  }
  render();
}

function resetFilters() {
  filterDefinitions.forEach(([id]) => { document.getElementById(id).value = ""; });
  document.getElementById("evidenceFilter").value = "";
  document.getElementById("metricFilter").value = "";
  render();
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
    changing: ["Platform, phase, workload, or experiment condition"],
    constants: ["comparability", "unit"],
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
      conditionIds: uniqueConditionIds(group),
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
    changing: (group) => varyingLabels(group, ["platform", "comparability", "experiment", "workload", "mode"]),
    constants: ["platform", "comparability", "workload", "mode"],
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
      platform, comparability, experiment, workload, mode,
      conditionId: conditionId(group[0]),
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
    if (options.changing) {
      const changing = typeof options.changing === "function" ? options.changing(group) : options.changing;
      container.append(comparisonContract(comparisonContext(group, changing, options.constants || [])));
    }
    const top = max(group.map((entry) => entry.value));
    group.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      setConditionTarget(row, entry.conditionId, entry.conditionIds);
      row.innerHTML = '<div class="bar-meta"><strong>' + escapeHtml(entry.label) + "</strong><span>" + escapeHtml(options.value(entry)) + "</span></div>" + (entry.detail ? '<div class="bar-detail">' + escapeHtml(entry.detail) + "</div>" : "") + '<div class="bar-track"><div class="bar-fill ' + (options.barClass || "") + '" style="width: ' + Math.max(3, (entry.value / top) * 100) + '%"></div></div>';
      container.append(row);
    });
  });
}

function renderRankingTable() {
  const result = filteredAnalysisRows(state.rankingSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "priority", "status", "mode", "strategy", "precision", "gpus"]);
  renderTable("rankingTable", result.rows, rankingColumns, result.message);
  document.getElementById("rankingCount").textContent = result.message || result.rows.length + " rows · select one for details";
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
      comparison: comparisonContext(group, ["Experiment condition"], ["platform_id", "comparability", "workload", "throughput_unit"]),
      points: group.map((row) => ({
        label: row.experiment_id + " · " + row.workload,
        value: number(row.throughput_tokens_mean),
        low: number(row.throughput_tokens_ci95_low) || number(row.throughput_tokens_mean),
        high: number(row.throughput_tokens_ci95_high) || number(row.throughput_tokens_mean),
        conditionId: conditionId(row),
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
  const card = chartCard(group.title, group.subtitle, group.comparison);
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
  const marks = group.points.map((point, index) => '<text class="svg-label" x="' + (left - 7) + '" y="' + (y(index) + 3) + '" text-anchor="end">' + escapeSvg(point.label) + '</text><line class="svg-line" stroke="' + colors[index % colors.length] + '" x1="' + scale(point.low) + '" y1="' + y(index) + '" x2="' + scale(point.high) + '" y2="' + y(index) + '"/><circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="' + colors[index % colors.length] + '" cx="' + scale(point.value) + '" cy="' + y(index) + '" r="4"><title>' + escapeSvg(point.label + ": " + formatCompact(point.value) + " (" + formatCompact(point.low) + "–" + formatCompact(point.high) + ")") + "</title></circle>").join("");
  card.append(svgElement(width, height, grids + marks));
  return card;
}

function renderStrategy() {
  const result = filteredAnalysisRows(state.strategySummary, ["platform_id", "comparability", "workload", "throughput_unit", "strategy", "precision", "gpus"]);
  const entries = buildStrategyEntries(result.rows);
  entries.forEach((entry) => {
    entry.conditionIds = matchingConditionIds({ platform_id: entry.platform, comparability: entry.comparability, workload: entry.workload, model: entry.model, precision: entry.precision, strategy: entry.strategy, gpus: entry.gpus, throughput_unit: entry.unit });
  });
  renderSeparatedBars("strategyBars", entries, { empty: result.message || "No completed distributed-strategy evidence.", value: (entry) => formatInteger(entry.value) + " " + entry.unit, changing: (group) => varyingLabels(group, ["platform", "comparability", "workload", "strategy", "precision", "gpus"]), constants: ["platform", "comparability", "workload", "precision", "gpus"] });
  document.getElementById("strategyStatus").textContent = result.message || entries.length + " conditions";
  markPanel("strategyBars", entries.length > 0, buildStrategyEntries(state.strategySummary).length > 0, "Distributed strategy", "strategy_summary.csv", true);
}

function buildStrategyEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.workload + " · " + row.strategy,
    detail: row.precision + " · " + row.gpus + " GPU" + (number(row.gpus) === 1 ? "" : "s"),
    platform: row.platform_id,
    comparability: row.comparability,
    workload: row.workload,
    model: row.model,
    strategy: row.strategy,
    precision: row.precision,
    gpus: row.gpus,
    unit: row.throughput_unit || "unit_not_recorded",
    value: number(row.avg_throughput_tokens_sec),
  })).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value);
}

function renderPrecision() {
  const result = filteredAnalysisRows(state.precisionSummary, ["platform_id", "comparability", "workload", "throughput_unit", "mode", "precision"]);
  const entries = buildPrecisionEntries(result.rows);
  entries.forEach((entry) => {
    entry.conditionIds = matchingConditionIds({ platform_id: entry.platform, comparability: entry.comparability, workload: entry.workload, model: entry.model, mode: entry.mode, precision: entry.precision, throughput_unit: entry.unit });
  });
  renderSeparatedBars("precisionBars", entries, { empty: result.message || "No completed precision-study evidence.", value: (entry) => formatInteger(entry.value) + " " + entry.unit, changing: (group) => varyingLabels(group, ["platform", "comparability", "workload", "precision", "mode"]), constants: ["platform", "comparability", "workload", "mode"] });
  document.getElementById("precisionStatus").textContent = result.message || entries.length + " conditions";
  markPanel("precisionBars", entries.length > 0, buildPrecisionEntries(state.precisionSummary).length > 0, "Precision trade-off", "precision_summary.csv", true);
}

function buildPrecisionEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.workload + " · " + row.precision,
    detail: row.mode + " · " + formatDecimal(number(row.avg_memory_used_gb), 2) + " GB",
    platform: row.platform_id,
    comparability: row.comparability,
    workload: row.workload,
    model: row.model,
    precision: row.precision,
    mode: row.mode,
    unit: row.throughput_unit || "unit_not_recorded",
    value: number(row.avg_throughput_tokens_sec),
  })).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value);
}

function renderMemory() {
  const result = filteredAnalysisRows(state.memorySummary, ["platform_id", "comparability", "workload", "throughput_unit"]);
  const microBatch = buildMemorySeries(result.rows);
  const checkpointing = buildCheckpointingGroups(result.rows);
  const accumulation = buildGradientAccumulationSeries(result.rows);
  renderLineCharts("memoryCharts", microBatch, { empty: result.message || "No completed micro-batch evidence.", xLabel: "micro-batch size", yLabel: "max memory (GB)" });
  renderCheckpointingCharts("checkpointingCharts", checkpointing, result.message || "No completed activation-checkpointing evidence.");
  renderLineCharts("accumulationCharts", accumulation, { empty: result.message || "No completed gradient-accumulation evidence.", xLabel: "gradient accumulation steps", yLabel: "max memory (GB)" });
  const capacity = microBatch.filter((item) => item.comparability === "capacity_adjusted");
  renderLineCharts("capacityCharts", capacity, { empty: result.message || "No completed capacity-adjusted L40S evidence.", xLabel: "micro-batch size", yLabel: "max memory (GB)" });
  document.getElementById("memoryStatus").textContent = result.message || microBatch.length + " micro-batch curves";
  document.getElementById("checkpointingStatus").textContent = result.message || checkpointing.length + " controlled comparisons";
  document.getElementById("accumulationStatus").textContent = result.message || accumulation.length + " accumulation curves";
  document.getElementById("capacityStatus").textContent = result.message || capacity.length + " L40S-only series";
  const allMemorySeries = buildMemorySeries(state.memorySummary);
  const allCheckpointing = buildCheckpointingGroups(state.memorySummary);
  const allAccumulation = buildGradientAccumulationSeries(state.memorySummary);
  markPanel("memoryCharts", microBatch.length > 0, allMemorySeries.length > 0, "Micro-batch study", "memory_summary.csv", true);
  markPanel("checkpointingCharts", checkpointing.length > 0, allCheckpointing.length > 0, "Activation checkpointing", "memory_summary.csv", true);
  markPanel("accumulationCharts", accumulation.length > 0, allAccumulation.length > 0, "Gradient accumulation", "memory_summary.csv", true);
  markPanel("capacityCharts", capacity.length > 0, allMemorySeries.some((item) => item.comparability === "capacity_adjusted"), "L40S capacity boundary", "memory_summary.csv", true);
}

function buildMemorySeries(rows) {
  const grouped = groupRows(rows.filter((row) => row.study_type === "micro_batch" && number(row.micro_batch_size) > 0 && number(row.max_memory_used_gb) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded", row.activation_checkpointing || "unspecified", row.gradient_accumulation_steps || "unspecified"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, unit, checkpointing, accumulation] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: "checkpointing: " + checkpointing + "; accumulation: " + accumulation + "; " + unit,
      platform, comparability, workload, unit,
      comparison: comparisonContext(group, ["Micro-batch size"], ["platform_id", "comparability", "workload", "activation_checkpointing", "gradient_accumulation_steps", "throughput_unit"]),
      points: group.map((row) => ({
        x: number(row.micro_batch_size),
        y: number(row.max_memory_used_gb),
        label: "batch " + row.micro_batch_size + ": " + formatDecimal(number(row.max_memory_used_gb), 2) + " GB; " + formatInteger(number(row.avg_throughput_tokens_sec)) + " " + unit + "; " + row.trials + " trials",
        conditionId: conditionId(row),
      })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function buildCheckpointingGroups(rows) {
  const grouped = groupRows(rows.filter((row) => row.study_type === "checkpointing" && number(row.max_memory_used_gb) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded", row.micro_batch_size || "unspecified", row.gradient_accumulation_steps || "unspecified"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, unit, microBatch, accumulation] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: "micro-batch: " + microBatch + "; accumulation: " + accumulation + "; " + unit,
      platform, comparability, workload, unit,
      comparison: comparisonContext(group, ["Activation checkpointing"], ["platform_id", "comparability", "workload", "micro_batch_size", "gradient_accumulation_steps", "throughput_unit"]),
      entries: group.map((row) => ({
        label: booleanLabel(row.activation_checkpointing, "Checkpointing on", "Checkpointing off"),
        value: number(row.max_memory_used_gb),
        throughput: number(row.avg_throughput_tokens_sec),
        unit,
        trials: number(row.trials),
        conditionId: conditionId(row),
      })).sort((left, right) => left.label.localeCompare(right.label)),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function buildGradientAccumulationSeries(rows) {
  const grouped = groupRows(rows.filter((row) => row.study_type === "gradient_accumulation" && number(row.gradient_accumulation_steps) > 0 && number(row.max_memory_used_gb) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded", row.activation_checkpointing || "unspecified", row.micro_batch_size || "unspecified"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, unit, checkpointing, microBatch] = key.split("\u0000");
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: "micro-batch: " + microBatch + "; checkpointing: " + checkpointing + "; " + unit,
      platform, comparability, workload, unit,
      comparison: comparisonContext(group, ["Gradient accumulation steps"], ["platform_id", "comparability", "workload", "micro_batch_size", "activation_checkpointing", "throughput_unit"]),
      points: group.map((row) => ({
        x: number(row.gradient_accumulation_steps),
        y: number(row.max_memory_used_gb),
        label: row.gradient_accumulation_steps + " accumulation steps: " + formatDecimal(number(row.max_memory_used_gb), 2) + " GB; " + formatInteger(number(row.avg_throughput_tokens_sec)) + " " + unit + "; " + row.trials + " trials",
        conditionId: conditionId(row),
      })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function booleanLabel(value, trueLabel, falseLabel) {
  return String(value).toLowerCase() === "true" ? trueLabel : falseLabel;
}

function renderCheckpointingCharts(containerId, groups, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState(emptyMessage));
    return;
  }
  groups.forEach((group) => container.append(checkpointingChartCard(group)));
}

function checkpointingChartCard(group) {
  const card = chartCard(group.title, group.subtitle, group.comparison);
  const bars = document.createElement("div");
  bars.className = "checkpointing-bars";
  const top = max(group.entries.map((entry) => entry.value));
  group.entries.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "bar-row checkpointing-row";
    setConditionTarget(row, entry.conditionId);
    const stateClass = entry.label.endsWith("on") ? "checkpointing-on" : "checkpointing-off";
    row.innerHTML = '<div class="bar-meta"><strong>' + escapeHtml(entry.label) + '</strong><span>' + formatDecimal(entry.value, 2) + ' GB</span></div>' +
      '<div class="bar-detail">' + formatInteger(entry.throughput) + " " + escapeHtml(entry.unit) + " · " + formatInteger(entry.trials) + ' trials</div>' +
      '<div class="bar-track"><div class="bar-fill ' + stateClass + '" style="width: ' + Math.max(3, (entry.value / top) * 100) + '%"></div></div>';
    bars.append(row);
  });
  card.append(bars);
  return card;
}

function renderScaling() {
  const result = filteredAnalysisRows(state.scalingSummary, ["platform_id", "comparability", "phase", "workload", "throughput_unit", "strategy", "precision", "gpus"]);
  const series = buildScalingSeries(result.rows);
  series.forEach((item) => item.points.forEach((point) => {
    point.conditionIds = matchingConditionIds({ platform_id: item.platform, comparability: item.comparability, workload: item.workload, strategy: item.strategy, precision: item.precision, scaling_type: item.scalingType, gpus: point.x, throughput_unit: item.unit });
  }));
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
      platform, comparability, workload, strategy, precision, unit, scalingType,
      comparison: comparisonContext(group, ["GPU count"], ["platform_id", "comparability", "workload", "strategy", "precision", "throughput_unit", "scaling_type"]),
      points: group.map((row) => ({ x: number(row.gpus), y: number(row.speedup), label: row.gpus + " GPU: " + formatDecimal(number(row.speedup), 2) + "× speedup; " + formatDecimal(number(row.scaling_efficiency) * 100, 1) + "% efficiency", conditionId: conditionId(row) })).sort((left, right) => left.x - right.x),
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
  const card = chartCard(series.title, series.subtitle, series.comparison);
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
  const points = series.points.map((point) => '<circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="' + color + '" cx="' + x(point.x) + '" cy="' + y(point.y) + '" r="4"><title>' + escapeSvg(point.label) + "</title></circle>").join("");
  const labels = '<text class="svg-label" x="' + width / 2 + '" y="' + (height - 3) + '" text-anchor="middle">' + escapeSvg(options.xLabel) + '</text><text class="svg-label" transform="translate(11 ' + height / 2 + ') rotate(-90)" text-anchor="middle">' + escapeSvg(options.yLabel) + "</text>";
  card.append(svgElement(width, height, grid + '<path class="svg-line" stroke="' + color + '" d="' + line + '"/>' + points + labels));
  return card;
}

function renderCommunication() {
  const result = filteredAnalysisRows(state.communicationSummary, ["platform_id", "comparability", "workload", "throughput_unit", "strategy", "gpus"]);
  const entries = buildCommunicationEntries(result.rows);
  renderSeparatedBars("communicationBars", entries, { empty: result.message || "No completed NCCL communication evidence.", barClass: "communication", value: (entry) => formatDecimal(entry.value, 2) + " GB/s", changing: (group) => varyingLabels(group, ["platform", "comparability", "workload", "strategy", "gpus"]), constants: ["platform", "comparability", "workload", "gpus"] });
  document.getElementById("communicationStatus").textContent = result.message || entries.length + " conditions";
  markPanel("communicationBars", entries.length > 0, buildCommunicationEntries(state.communicationSummary).length > 0, "NCCL communication", "communication_summary.csv", true);
}

function buildCommunicationEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.experiment_id + " · " + row.strategy,
    detail: row.workload + " · " + row.gpus + " GPUs · communication/compute " + formatDecimal(number(row.communication_to_compute_ratio), 2),
    conditionId: conditionId(row),
    platform: row.platform_id,
    comparability: row.comparability,
    workload: row.workload,
    strategy: row.strategy,
    gpus: row.gpus,
    unit: "GB/s",
    value: number(row.avg_nccl_bandwidth_gbps),
  })).filter((entry) => entry.value > 0).sort((left, right) => right.value - left.value);
}

function renderDataMovement() {
  const result = filteredAnalysisRows(state.dataMovementSummary, ["platform_id", "comparability", "workload", "throughput_unit"]);
  const entries = buildDataMovementEntries(result.rows);
  renderSeparatedBars("movementBars", entries, { empty: result.message || "No completed data-movement evidence.", value: (entry) => formatDecimal(entry.value, 3) + " s", changing: (group) => varyingLabels(group, ["platform", "comparability", "workload", "num_workers", "pinned_memory", "prefetch_factor"]), constants: ["platform", "comparability", "workload"] });
  document.getElementById("movementStatus").textContent = result.message || entries.length + " conditions";
  markPanel("movementBars", entries.length > 0, buildDataMovementEntries(state.dataMovementSummary).length > 0, "Data movement", "data_movement_summary.csv", true);
}

function buildDataMovementEntries(rows) {
  return rows.map((row) => ({
    label: row.platform_id + " · " + row.comparability + " · " + row.experiment_id + " · " + row.workload,
    detail: "workers " + row.num_workers + "; pinned memory " + row.pinned_memory + "; prefetch " + row.prefetch_factor,
    conditionId: conditionId(row),
    platform: row.platform_id,
    comparability: row.comparability,
    workload: row.workload,
    num_workers: row.num_workers,
    pinned_memory: row.pinned_memory,
    prefetch_factor: row.prefetch_factor,
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
      comparison: comparisonContext(group, varyingLabels(group, ["framework", "precision", "gpus", "batch_size", "concurrent_requests", "prompt_length"]), ["platform_id", "comparability", "workload", "throughput_unit"]),
      points: group.map((row) => ({ x: number(row.avg_latency_p95_ms), y: number(row.avg_throughput_tokens_sec), label: row.experiment_id + " · " + row.workload + " · " + row.precision + " · " + row.gpus + " GPU", conditionId: conditionId(row) })),
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
  const card = chartCard(group.title, group.subtitle, group.comparison);
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
  const points = group.points.map((point, index) => '<circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="' + colors[index % colors.length] + '" cx="' + x(point.x) + '" cy="' + y(point.y) + '" r="5"><title>' + escapeSvg(point.label + ": p95 " + formatCompact(point.x) + " ms; " + formatCompact(point.y) + " " + group.unit) + "</title></circle>").join("");
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
  const studySections = ["research", "overview", "performance", "memory", "scaling", "inference", "evidence"];

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
    setConditionTarget(tr, conditionId(row));
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

function openExperimentDialog(requestedIds) {
  const ids = [...new Set(requestedIds.map((value) => value.trim()).filter(Boolean))];
  const available = ids.filter((id) => rowsForCondition(id).length);
  if (!available.length) return;
  renderExperimentDialog(available[0], available);
  const dialog = document.getElementById("experimentDialog");
  if (!dialog.open) dialog.showModal();
}

function closeExperimentDialog() {
  const dialog = document.getElementById("experimentDialog");
  if (dialog.open) dialog.close();
}

function rowsForCondition(id) {
  return state.results.filter((row) => conditionId(row) === id || (!id.includes("@") && row.experiment_id === id));
}

function renderExperimentDialog(selectedId, availableIds) {
  const detail = buildExperimentDetail(rowsForCondition(selectedId));
  if (!detail) return;
  document.getElementById("experimentDialogTitle").textContent = detail.experimentId + " · " + detail.experimentName;
  document.getElementById("experimentDialogSubtitle").textContent = [detail.platform, detail.comparability, detail.workload, detail.mode].filter(Boolean).join(" · ");
  const body = document.getElementById("experimentDialogBody");
  body.replaceChildren();

  if (availableIds.length > 1) {
    const switcher = document.createElement("div");
    switcher.className = "condition-switcher";
    const label = document.createElement("span");
    label.textContent = "Related evidence";
    switcher.append(label);
    availableIds.forEach((id) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = id;
      button.className = id === selectedId ? "is-active" : "";
      button.addEventListener("click", () => renderExperimentDialog(id, availableIds));
      switcher.append(button);
    });
    body.append(switcher);
  }

  const summary = document.createElement("section");
  summary.className = "detail-hero";
  summary.innerHTML = '<div><span class="status-pill status-' + escapeHtml(detail.status) + '">' + escapeHtml(detail.status) + '</span><p>' + escapeHtml(detail.purpose) + '</p></div><dl>' +
    detail.summary.map((item) => '<div><dt>' + escapeHtml(item.label) + '</dt><dd>' + escapeHtml(item.value) + "</dd></div>").join("") + "</dl>";
  body.append(summary);

  const protocol = document.createElement("section");
  protocol.className = "detail-section";
  protocol.innerHTML = '<div class="detail-section-heading"><div><p class="eyebrow">Protocol</p><h3>What this condition changes and controls</h3></div></div>' +
    '<div class="detail-protocol"><div><span>Study variable</span><strong>' + escapeHtml(detail.studyVariable) + '</strong></div><div><span>Recorded controls</span><div class="detail-chip-list">' +
    detail.controls.map((item) => '<span>' + escapeHtml(item.label + ": " + item.value) + "</span>").join("") + "</div></div></div>";
  body.append(protocol);

  const metricsGrid = document.createElement("div");
  metricsGrid.className = "detail-section-grid";
  detail.sections.forEach((section) => metricsGrid.append(detailMetricSection(section, detail)));
  body.append(metricsGrid);

  const trials = document.createElement("section");
  trials.className = "detail-section detail-trials";
  const trialMetric = selectTrialMetric(detail.rows);
  trials.innerHTML = '<div class="detail-section-heading"><div><p class="eyebrow">Evidence</p><h3>Trial-level consistency</h3></div><span>' + detail.rows.length + " recorded trial" + (detail.rows.length === 1 ? "" : "s") + "</span></div>" + renderTrialBars(detail.rows, trialMetric);
  body.append(trials);

  const provenance = document.createElement("details");
  provenance.className = "detail-provenance";
  provenance.innerHTML = '<summary>Environment and provenance</summary><dl>' + detail.provenance.map((item) => '<div><dt>' + escapeHtml(item.label) + '</dt><dd>' + escapeHtml(item.value) + "</dd></div>").join("") + "</dl>";
  body.append(provenance);
}

function buildExperimentDetail(rows) {
  if (!rows.length) return null;
  const first = rows[0];
  const completed = rows.filter((row) => row.status === "completed");
  const metricRows = completed.length ? completed : rows;
  const status = completed.length === rows.length ? "completed" : completed.length ? "partial" : (first.status || "unknown");
  const sections = detailMetricSections.map(([title, definitions]) => ({
    title,
    items: definitions.map(([label, keys, unit, digits]) => metricAggregate(metricRows, label, keys, unit === "throughput_unit" ? first.throughput_unit : unit, digits)).filter(Boolean),
  })).filter((section) => section.items.length);
  const controls = [
    ["Platform", "platform_id"], ["Comparability", "comparability"], ["Workload", "workload"], ["Model", "model"], ["Mode", "mode"],
    ["Precision", "precision"], ["Strategy", "strategy"], ["GPUs", "gpus"], ["Batch size", "batch_size"], ["Global batch", "global_batch_size"],
    ["Per-GPU batch", "per_gpu_batch_size"], ["Sequence length", "parameter_sequence_length"], ["Checkpointing", "parameter_activation_checkpointing"],
    ["Gradient accumulation", "parameter_gradient_accumulation_steps"], ["Framework", "framework"], ["Concurrency", "parameter_concurrent_requests"],
  ].flatMap(([label, key]) => hasValue(first[key]) ? [{ label, value: String(first[key]) }] : []);
  const summary = [
    { label: "Condition", value: conditionId(first) },
    { label: "Trials", value: completed.length + "/" + rows.length + " completed" },
    { label: "Research questions", value: researchQuestionsForRow(first).join(", ") || "Not assigned" },
    { label: "Key metrics", value: readableList(first.key_metrics) || "Recorded metrics below" },
    { label: "Wave", value: rowValue(first, "wave") || "Not recorded" },
  ];
  const provenance = [
    ["Run IDs", rows.map((row) => row.run_id).filter(Boolean).join(", ")],
    ["Git commit", first.git_commit || first.environment_git_commit],
    ["GPU", first.observed_gpu_model || first.environment_observed_gpu_model || first.gpu_model],
    ["GPU UUID mapping", first.gpu_telemetry_mapping || first.environment_observed_gpu_uuid_mapping_method],
    ["Profiled GPU UUIDs", first.profiled_gpu_uuids || first.metric_nvidia_smi_profiled_gpu_uuids],
    ["CUDA", first.environment_cuda_version], ["PyTorch", first.environment_torch_version], ["vLLM", first.environment_vllm_version],
    ["NCCL", first.environment_nccl_version], ["Slurm job", first.environment_slurm_job_id], ["Source", first.source_path],
  ].flatMap(([label, value]) => hasValue(value) ? [{ label, value: String(value) }] : []);
  return {
    rows,
    status,
    sections,
    controls,
    summary,
    provenance,
    experimentId: first.experiment_id || "Unknown experiment",
    experimentName: String(first.experiment || first.phase || "Recorded condition").replaceAll("_", " "),
    purpose: first.objective || "Recorded evidence for " + (first.phase || "the experiment matrix") + ". Open the metric groups below to inspect the measured condition.",
    studyVariable: first.parameter_comparison_axis || first.parameter_scaling_dimension || first.parameter_memory_study_type || first.parameter_inference_study_type || "Defined by the experiment matrix; inspect the controls below when comparing with another condition.",
    platform: first.platform_id,
    comparability: first.comparability,
    workload: first.workload,
    mode: first.mode,
    energyScope: String(first.energy_scope || first.metric_energy_scope || "").replaceAll("_", " "),
  };
}

function metricAggregate(rows, label, keys, unit, digits) {
  const values = rows.flatMap((row) => {
    const value = firstRecorded(row, keys);
    return hasValue(value) && Number.isFinite(Number.parseFloat(value)) ? [Number.parseFloat(value)] : [];
  });
  if (!values.length) return null;
  return { label, value: average(values), min: Math.min(...values), max: Math.max(...values), unit: unit || "", digits, samples: values.length };
}

function firstRecorded(row, keys) {
  const key = keys.find((candidate) => hasValue(row[candidate]));
  return key ? row[key] : undefined;
}

function hasValue(value) {
  return value !== "" && value !== undefined && value !== null;
}

function readableList(value) {
  if (!hasValue(value)) return "";
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) return parsed.map((item) => String(item).replaceAll("_", " ")).join(", ");
  } catch {
    // Fall through to the plain string representation.
  }
  return String(value).replaceAll("_", " ");
}

function detailMetricSection(section, detail) {
  const element = document.createElement("section");
  element.className = "detail-section metric-section";
  const gauge = detailGauge(section, detail);
  const scope = section.title === "Power and energy" ? '<span>' + escapeHtml(detail.energyScope || "Scope not recorded") + "</span>" : "";
  element.innerHTML = '<div class="detail-section-heading"><h3>' + escapeHtml(section.title) + "</h3>" + scope + '</div><div class="metric-card-grid">' + section.items.map((item) => {
    const range = item.samples > 1 && item.min !== item.max ? '<small>range ' + formatMetric(item.min, item) + "–" + formatMetric(item.max, item) + "</small>" : '<small>' + item.samples + " sample" + (item.samples === 1 ? "" : "s") + "</small>";
    return '<article><span>' + escapeHtml(item.label) + '</span><strong>' + escapeHtml(formatMetric(item.value, item)) + "</strong>" + range + "</article>";
  }).join("") + "</div>" + gauge;
  return element;
}

function detailGauge(section, detail) {
  if (section.title === "Memory") {
    const used = section.items.find((item) => item.label === "GPU memory used");
    const total = average(detail.rows.map((row) => Number.parseFloat(row.environment_observed_gpu_memory_gb)).filter(Number.isFinite));
    if (used && total > 0) return gaugeMarkup("Observed memory pressure", used.value, total, "GB");
  }
  if (section.title === "GPU telemetry") {
    const utilization = section.items.find((item) => item.label === "Average GPU utilisation");
    if (utilization) return gaugeMarkup("Average GPU utilisation", utilization.value, 100, "%");
  }
  return "";
}

function gaugeMarkup(label, value, maxValue, unit) {
  const bounded = Math.max(0, Math.min(100, value / maxValue * 100));
  return `<div class="detail-gauge"><div><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatDecimal(value, 1) + " " + unit)}</strong></div><div class="detail-gauge-track"><span style="width:${bounded}%"></span></div><small>${formatDecimal(bounded, 1)}% of ${formatDecimal(maxValue, 1)} ${escapeHtml(unit)}</small></div>`;
}

function formatMetric(value, item) {
  return formatDecimal(value, item.digits) + (item.unit ? " " + item.unit : "");
}

function selectTrialMetric(rows) {
  const candidates = [
    { label: "Throughput", keys: ["throughput_tokens_sec", "metric_throughput_tokens_sec"], unit: rows[0].throughput_unit || "" },
    { label: "Runtime", keys: ["runtime_seconds"], unit: "s" },
    { label: "GPU memory", keys: ["metric_nvidia_smi_memory_used_gb_measured_region", "memory_used_gb"], unit: "GB" },
  ];
  return candidates.find((candidate) => rows.some((row) => hasValue(firstRecorded(row, candidate.keys)))) || candidates[1];
}

function renderTrialBars(rows, metric) {
  const values = rows.map((row) => Number.parseFloat(firstRecorded(row, metric.keys))).filter(Number.isFinite);
  const top = values.length ? Math.max(...values) : 1;
  return '<div class="trial-bars">' + rows.slice().sort((left, right) => number(left.repetition_number) - number(right.repetition_number)).map((row) => {
    const raw = Number.parseFloat(firstRecorded(row, metric.keys));
    const value = Number.isFinite(raw) ? raw : 0;
    const error = row.error ? '<small title="' + escapeHtml(row.error) + '">' + escapeHtml(row.error) + "</small>" : "";
    return '<article><div><strong>Trial ' + escapeHtml(row.repetition_number || "—") + '</strong><span class="status-' + escapeHtml(row.status || "unknown") + '">' + escapeHtml(row.status || "unknown") + '</span><b>' + escapeHtml(value ? formatDecimal(value, 2) + " " + metric.unit : "No " + metric.label.toLowerCase()) + '</b></div><div class="trial-bar-track"><span style="width:' + (top ? value / top * 100 : 0) + '%"></span></div>' + error + "</article>";
  }).join("") + "</div>";
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

function chartCard(title, subtitle, comparison) {
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
  if (comparison) card.append(comparisonContract(comparison));
  return card;
}

function comparisonContext(rows, changing, constantFields) {
  const constants = constantFields.flatMap((key) => {
    const values = [...new Set(rows.map((row) => String(row[key] || "").trim()).filter(Boolean))];
    return values.length === 1 ? [{ label: displayName(key), value: values[0] }] : [];
  });
  return { changing: changing.length ? changing : ["Multiple recorded settings"], constants };
}

function comparisonContract(context) {
  const element = document.createElement("div");
  element.className = "comparison-contract";
  const changing = document.createElement("div");
  changing.className = "comparison-changing";
  changing.innerHTML = "<span>Changing</span><strong>" + escapeHtml(context.changing.join(" · ")) + "</strong>";
  element.append(changing);
  if (context.constants.length) {
    const constants = document.createElement("div");
    constants.className = "comparison-constants";
    const label = document.createElement("span");
    label.textContent = "Held constant";
    constants.append(label);
    context.constants.forEach((item) => {
      const chip = document.createElement("span");
      chip.className = "comparison-chip";
      chip.textContent = item.label + ": " + item.value;
      constants.append(chip);
    });
    element.append(constants);
  }
  return element;
}

function varyingLabels(rows, fields) {
  const labels = fields.filter((key) => new Set(rows.map((row) => String(row[key] || "").trim()).filter(Boolean)).size > 1).map(displayName);
  return labels.length ? labels : ["Experiment condition"];
}

function conditionId(row) {
  return row.condition_id || (row.experiment_id && row.platform_id ? row.experiment_id + "@" + row.platform_id : "");
}

function uniqueConditionIds(rows) {
  return [...new Set(rows.map(conditionId).filter(Boolean))];
}

function matchingConditionIds(criteria) {
  const entries = Object.entries(criteria).filter(([, value]) => hasValue(value));
  return uniqueConditionIds(state.results.filter((row) => row.status === "completed" && entries.every(([key, value]) => String(row[key] || "") === String(value))));
}

function parseConditionIds(value) {
  return String(value || "").split(/[;,]/).map((item) => item.trim()).filter(Boolean);
}

function setConditionTarget(element, id, ids) {
  const targets = id ? [id] : (ids || []);
  if (!targets.length) return;
  element.dataset.conditionIds = targets.join(",");
  element.classList.add("experiment-target");
  element.tabIndex = 0;
  element.setAttribute("role", "button");
  element.setAttribute("aria-label", "View experiment details");
}

function conditionTargetAttributes(point) {
  const targets = point.conditionId ? [point.conditionId] : (point.conditionIds || []);
  if (!targets.length) return "";
  return 'tabindex="0" role="button" aria-label="View experiment details" data-condition-ids="' + escapeSvg(targets.join(",")) + '"';
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
    buildCheckpointingGroups,
    buildCompletionEntries,
    buildEnergyPowerEntries,
    buildExperimentDetail,
    buildGradientAccumulationSeries,
    buildInferenceGroups,
    buildMemorySeries,
    buildPhaseBarEntries,
    buildRq1ScalingSeries,
    buildRq2StrategyGroups,
    buildScalingSeries,
    buildUncertaintyGroups,
    comparisonContext,
    summarizePhases,
  };
}
