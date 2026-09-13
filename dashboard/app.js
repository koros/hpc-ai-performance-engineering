const state = {
  results: [],
  trialSummary: [],
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
  loading: true,
  loadErrors: [],
  runtimeError: "",
  sort: { table: "trialTable", key: "experiment_id", direction: 1 },
};

const paths = {
  results: "../results/processed/experiments.csv",
  trialSummary: "../results/analysis/trial_summary.csv",
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
};

const analysisFileKeys = {
  "trial_summary.csv": "trialSummary",
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

const trialColumns = ["experiment_id", "platform_id", "comparability", "workload", "completed_trials", "failed_trials", "evidence_status", "throughput_tokens_mean", "throughput_tokens_ci95_low", "throughput_tokens_ci95_high"];
const colors = ["#0f766e", "#2563eb", "#7c3aed", "#b45309", "#be123c", "#0891b2", "#4d7c0f"];
const scalingColors = [
  "#2563eb", "#7c3aed", "#be123c", "#c2410c", "#a16207", "#4d7c0f", "#047857",
  "#0f766e", "#0e7490", "#0369a1", "#4338ca", "#a21caf", "#9f1239", "#475569",
];
const strategyColors = {
  ddp: "#2563eb",
  fsdp: "#0f766e",
  fsdp_full_shard: "#0f766e",
  fsdp_sharded_grad_op: "#14b8a6",
  deepspeed_zero1: "#7c3aed",
  deepspeed_zero2: "#d97706",
  deepspeed_zero3: "#be123c",
};
const precisionColors = {
  fp32: "#475569",
  fp16: "#7c3aed",
  bf16: "#0f766e",
  fp8: "#e11d48",
};
const hardwareColors = {
  h100: "#7c3aed",
  l40s: "#0f766e",
};
const hardwareProtocolFields = [
  "phase", "workload", "model", "mode", "gpus", "strategy", "scaling_type", "framework",
  "batch_size", "global_batch_size", "per_gpu_batch_size", "precision", "throughput_unit",
  "parameter_activation_checkpointing", "parameter_allreduce_bytes", "parameter_base_compute_dtype",
  "parameter_baseline_kind", "parameter_batch_size", "parameter_comparison_axis",
  "parameter_concurrent_requests", "parameter_data_movement_study_type", "parameter_data_parallel_size",
  "parameter_distributed_strategy", "parameter_effective_batch_size", "parameter_framework",
  "parameter_framework_label", "parameter_gradient_accumulation_steps", "parameter_inference_study_type",
  "parameter_max_new_tokens", "parameter_memory_study_type", "parameter_micro_batch_size",
  "parameter_num_workers", "parameter_per_gpu_batch_size", "parameter_pinned_memory",
  "parameter_precision_backend", "parameter_prefetch_factor", "parameter_profile_target",
  "parameter_prompt_length", "parameter_quantization_method", "parameter_repeat_index",
  "parameter_scaling_type", "parameter_sequence_length", "parameter_stack", "parameter_strategy_detail",
  "parameter_tensor_parallel_size",
];
const researchTargets = {
  RQ1: "research-scaling",
  RQ2: "research-strategy",
  RQ3: "precisionBars",
  RQ4: "scaling",
  RQ5: "energyCoverage",
  RQ6: "hardwareComparison",
  RQ7: "evidence",
};

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
    safeRender();
    loadDefaultData().catch((error) => {
      state.loading = false;
      state.loadErrors.push("Dashboard startup: " + errorMessage(error));
      updateDashboardAlert();
    });
  });
}

function bindInputs() {
  filterDefinitions.forEach(([id]) => document.getElementById(id).addEventListener("change", safeRender));
  document.getElementById("evidenceFilter").addEventListener("change", safeRender);
  document.getElementById("metricFilter").addEventListener("change", safeRender);
  document.getElementById("trainingStudyFilter").addEventListener("change", safeRender);
  document.getElementById("strategyComparisonFilter").addEventListener("change", safeRender);
  document.getElementById("strategyMetricFilter").addEventListener("change", safeRender);
  document.getElementById("precisionComparisonFilter").addEventListener("change", safeRender);
  document.getElementById("hardwarePhaseFilter").addEventListener("change", safeRender);
  document.getElementById("hardwareWorkloadFilter").addEventListener("change", safeRender);
  document.getElementById("hardwareModeFilter").addEventListener("change", safeRender);
  document.getElementById("hardwareComparisonFilter").addEventListener("change", safeRender);
  document.getElementById("scalingTypeFilter").addEventListener("change", safeRender);
  document.getElementById("resetFilters").addEventListener("click", resetFilters);
  document.getElementById("resultsFile").addEventListener("change", (event) => {
    readFile(event.target.files[0]).then((text) => {
      state.results = parseCsv(text);
      rebuildFilters();
      safeRender();
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
    safeRender();
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
  state.loadErrors = [];
  const entries = Object.entries(paths);
  const loaded = await Promise.all(entries.map(async ([key, path]) => [key, await fetchCsv(path, key)]));
  loaded.forEach(([key, rows]) => {
    state[key] = rows;
  });
  state.loading = false;
  rebuildFilters();
  safeRender();
}

async function fetchCsv(path, label) {
  try {
    const response = await fetch(path, { cache: "no-store" });
    if (response.ok) return parseCsv(await response.text());
    state.loadErrors.push(label + ": HTTP " + response.status);
  } catch (error) {
    state.loadErrors.push(label + ": " + errorMessage(error));
  }
  return [];
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function safeRender() {
  try {
    render();
    state.runtimeError = "";
  } catch (error) {
    state.runtimeError = errorMessage(error);
    console.error("Dashboard rendering failed", error);
  }
  updateDashboardAlert();
}

function updateDashboardAlert() {
  const alert = document.getElementById("dashboardAlert");
  const message = dashboardAlertMessage({
    loading: state.loading,
    loadErrors: state.loadErrors,
    runtimeError: state.runtimeError,
    resultCount: state.results.length,
    protocol: window.location.protocol,
  });
  alert.textContent = message;
  alert.hidden = !message;
}

function dashboardAlertMessage({ loading, loadErrors, runtimeError, resultCount, protocol }) {
  if (runtimeError) return "Dashboard rendering stopped: " + runtimeError + ". Check the browser console for details.";
  if (loading) return "";
  if (protocol === "file:") return "Results cannot be loaded from a local file URL. Serve the repository root with make serve-dashboard, then open the displayed HTTP address.";
  if (loadErrors.length) return "Some dashboard evidence could not be loaded: " + loadErrors.join("; ") + ".";
  if (!resultCount) return "No experiment rows were loaded. Refresh the analysis outputs or select a results CSV manually.";
  return "";
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
  rebuildTrainingStudyFilter();
}

function rebuildTrainingStudyFilter() {
  const select = document.getElementById("trainingStudyFilter");
  const current = select.value;
  const studies = uniqueValues(
    state.trialSummary.filter((row) => row.mode === "training" && number(row.throughput_tokens_mean) > 0),
    "phase",
  );
  select.replaceChildren();
  studies.forEach((study) => select.append(option(study, study)));
  const preferred = studies.includes(current)
    ? current
    : studies.includes("Strong scaling")
      ? "Strong scaling"
      : studies[0] || "";
  select.value = preferred;
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
  renderResearchSnapshot(rows);
  renderDecisionBrief();
  renderResearchQuestions();
  renderKpis(rows);
  renderProgress(rows);
  renderStudyMap(rows);
  renderEnergy(rows);
  renderUncertainty();
  renderStrategy();
  renderPrecision();
  renderHardware();
  renderMemory();
  renderScaling();
  renderCommunication();
  renderDataMovement();
  renderInference();
  renderTrialTable();
  organizeSections();
}

function renderResearchSnapshot(rows) {
  const completed = rows.filter((row) => row.status === "completed");
  const conditions = uniqueConditionIds(rows);
  const research = buildResearchQuestionOverview(state.researchQuestionSummary);
  const ready = research.filter((entry) => entry.status === "ready");
  const partial = research.filter((entry) => entry.status === "partial");
  const fp8Rows = completed.filter((row) => String(row.precision).toLowerCase() === "fp8");
  const fp8Conditions = uniqueConditionIds(fp8Rows);
  const measuredEnergy = completed.filter((row) => row.energy_scope === "measured_region" && row.energy_measurement_status === "sufficient");
  const legacyTelemetry = completed.filter((row) => row.gpu_telemetry_mapping === "legacy_positional");
  const platforms = uniqueValues(rows, "platform_id").map(displayPlatform);
  const filtered = Object.values(selectedFilters()).some(Boolean) || Boolean(document.getElementById("metricFilter").value);

  document.getElementById("heroTrialCount").textContent = formatInteger(completed.length);
  document.getElementById("heroReadyQuestions").textContent = ready.length + "/" + research.length;
  document.getElementById("heroSummary").textContent = partial.length
    ? ready.length + " research questions are answer-ready. " + partial.map((entry) => entry.researchQuestion).join(", ") + " retains a clearly labelled evidence limitation."
    : "Every research question represented in the current analysis has sufficient direct evidence.";
  document.getElementById("researchReadinessDial").style.setProperty("--readiness", research.length ? (ready.length / research.length) * 360 + "deg" : "0deg");

  const scope = document.getElementById("snapshotScope");
  scope.replaceChildren();
  [filtered ? "Filtered view" : "Complete evidence view", platforms.join(" + ") || "No platform", formatInteger(conditions.length) + " conditions"].forEach((label) => {
    const chip = document.createElement("span");
    chip.textContent = label;
    scope.append(chip);
  });

  const highlights = [
    {
      label: "Evidence coverage",
      value: formatInteger(conditions.length) + " conditions",
      detail: formatInteger(completed.length) + " completed trials in the active view",
      tone: "complete",
    },
    {
      label: "Validated FP8 training",
      value: fp8Conditions.length ? formatInteger(fp8Conditions.length) + " conditions" : "Not in this view",
      detail: fp8Rows.length ? formatInteger(fp8Rows.length) + " trials across " + uniqueValues(fp8Rows, "platform_id").map(displayPlatform).join(" and ") : "Adjust filters to inspect FP8 evidence",
      tone: fp8Rows.length ? "new" : "muted",
    },
    {
      label: "Energy-ready evidence",
      value: formatInteger(measuredEnergy.length) + " trials",
      detail: "Measured-region records with sufficient samples",
      tone: measuredEnergy.length ? "complete" : "muted",
    },
    {
      label: "Telemetry boundary",
      value: formatInteger(legacyTelemetry.length) + " legacy rows",
      detail: "Retained for non-energy metrics; legacy power and energy are excluded",
      tone: legacyTelemetry.length ? "caution" : "complete",
    },
  ];
  const container = document.getElementById("campaignHighlights");
  container.replaceChildren();
  highlights.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "campaign-highlight highlight-" + entry.tone;
    card.innerHTML = "<span>" + escapeHtml(entry.label) + "</span><strong>" + escapeHtml(entry.value) + "</strong><small>" + escapeHtml(entry.detail) + "</small>";
    container.append(card);
  });
}

function buildResearchQuestionOverview(rows) {
  return Object.entries(groupRows(rows, (row) => [row.research_question])).map(([researchQuestion, group]) => {
    const statuses = group.map((row) => row.evidence_status);
    const positive = statuses.every((status) => status === "sufficient" || status === "supporting_platform_evidence");
    const pending = statuses.every((status) => status.startsWith("awaiting") || status === "insufficient_data");
    const status = positive ? "ready" : pending ? "pending" : "partial";
    const completedTrials = sum(group.map((row) => number(row.completed_trials)));
    const requiredTrials = sum(group.map((row) => number(row.required_trials)));
    return {
      researchQuestion,
      objective: group[0].objective,
      question: group[0].question,
      status,
      statusLabel: status === "ready" ? "Answer-ready" : status === "pending" ? "Awaiting evidence" : researchQuestion === "RQ5" ? "Partial energy coverage" : "Evidence limitation",
      conditions: sum(group.map((row) => number(row.direct_conditions))),
      completedTrials,
      requiredTrials,
      energyTrials: sum(group.map((row) => number(row.energy_trials))),
      platforms: uniqueValues(group, "platform_id").map(displayPlatform),
      comparabilities: uniqueValues(group, "comparability").map(displayComparability),
      limitation: group.find((row) => row.evidence_status !== "sufficient")?.scope_note || group[0].limitation,
      target: researchTargets[researchQuestion] || "research",
    };
  }).sort((left, right) => left.researchQuestion.localeCompare(right.researchQuestion));
}

function buildDecisionInsights({ researchFindings = [], hardware = [], communication = [], memory = [], results = [] }) {
  return [
    buildScalingDecision(researchFindings),
    buildStrategyDecision(researchFindings),
    buildPrecisionDecision(results),
    buildMemoryDecision(memory),
    buildCommunicationDecision(communication),
    buildHardwareDecision(hardware),
    buildEnergyDecision(results),
  ].filter(Boolean);
}

function renderDecisionBrief() {
  const allInsights = buildDecisionInsights({
    researchFindings: state.researchFindingSummary,
    hardware: state.hardwareSummary,
    communication: state.communicationSummary,
    memory: state.memorySummary,
    results: state.results,
  });
  const filters = selectedFilters();
  const insights = allInsights.filter((insight) => {
    if (filters.research_question && insight.researchQuestion !== filters.research_question) return false;
    if (filters.platform_id && !insight.platforms.includes(filters.platform_id)) return false;
    if (filters.workload && !insight.workloads.includes(filters.workload)) return false;
    return true;
  });
  const container = document.getElementById("decisionInsightGrid");
  container.replaceChildren();
  document.getElementById("decisionBriefStatus").textContent = insights.length
    ? insights.length + " evidence-led conclusion" + (insights.length === 1 ? "" : "s")
    : "No conclusion matches the active scope";
  renderDecisionBriefSummary(allInsights, insights);
  if (!insights.length) {
    container.append(emptyState("No campaign conclusion matches the selected platform, workload, or research question. Clear those filters to restore the complete synthesis."));
    return;
  }
  insights.forEach((insight) => container.append(decisionInsightCard(insight)));
}

function renderDecisionBriefSummary(allInsights, visibleInsights) {
  const container = document.getElementById("decisionBriefSummary");
  const hardware = allInsights.find((insight) => insight.id === "hardware");
  const memory = allInsights.find((insight) => insight.id === "memory");
  const energy = allInsights.find((insight) => insight.id === "energy");
  if (!visibleInsights.length) {
    container.replaceChildren();
    return;
  }
  const recommendations = [
    hardware ? ["Training", "Prefer H100 for the matched training protocols"] : null,
    hardware ? ["Inference", "Benchmark by workload; the training advantage does not transfer automatically"] : null,
    memory ? ["Memory pressure", "Use checkpointing for capacity, accepting its measured throughput cost"] : null,
    energy && energy.tone === "limitation" ? ["Energy", "Keep the final hardware energy conclusion open until E235–E238"] : null,
  ].filter(Boolean);
  container.innerHTML = '<div class="decision-thesis"><span>Current overall interpretation</span><strong>No single configuration wins every objective.</strong><p>Hardware, workload, execution mode, and memory constraints change the answer. The recommendations below are conditional on matched experimental controls.</p></div>' +
    '<div class="decision-rules">' + recommendations.map(([label, value]) => '<div><span>' + escapeHtml(label) + '</span><strong>' + escapeHtml(value) + '</strong></div>').join("") + "</div>";
}

function decisionInsightCard(insight) {
  const card = document.createElement("article");
  card.className = "decision-insight insight-" + insight.tone;
  const evidenceButton = insight.conditionIds.length
    ? '<button type="button" class="insight-evidence" data-condition-ids="' + escapeHtml(insight.conditionIds.join(",")) + '">Inspect ' + formatInteger(insight.conditionIds.length) + " source condition" + (insight.conditionIds.length === 1 ? "" : "s") + "</button>"
    : '<span class="insight-evidence-note">Evidence coverage statement</span>';
  card.innerHTML = '<div class="insight-heading"><span class="insight-rq">' + escapeHtml(insight.researchQuestion) + '</span><span class="insight-confidence">' + escapeHtml(insight.confidence) + "</span></div>" +
    "<h3>" + escapeHtml(insight.title) + "</h3>" +
    '<strong class="insight-metric">' + escapeHtml(insight.metric) + "</strong>" +
    '<p class="insight-statement">' + escapeHtml(insight.statement) + "</p>" +
    '<p class="insight-implication"><span>Decision:</span> ' + escapeHtml(insight.implication) + "</p>" +
    '<dl class="insight-contract"><div><dt>What changed</dt><dd>' + escapeHtml(insight.changed) + "</dd></div><div><dt>What stayed fixed</dt><dd>" + escapeHtml(insight.heldConstant) + "</dd></div></dl>" +
    '<p class="insight-limitation"><span>Boundary</span>' + escapeHtml(insight.limitation) + "</p>" +
    '<footer>' + evidenceButton + '<a href="#' + escapeHtml(insight.target) + '">See supporting analysis <span aria-hidden="true">→</span></a></footer>';
  return card;
}

function buildScalingDecision(rows) {
  const exact = rows.filter((row) => row.research_question === "RQ1" && row.finding_status === "sufficient" && row.comparability === "exact" && number(row.gpus) >= 4);
  if (!exact.length) return null;
  const target = exact.some((row) => row.platform_id === "h100") ? exact.filter((row) => row.platform_id === "h100") : exact;
  const highestGpuCount = max(target.map((row) => number(row.gpus)));
  const candidates = target.filter((row) => number(row.gpus) === highestGpuCount).sort((left, right) => number(right.scaling_efficiency) - number(left.scaling_efficiency));
  const best = candidates[0];
  const weakest = candidates[candidates.length - 1];
  const efficiency = number(best.scaling_efficiency) * 100;
  const range = weakest && weakest !== best
    ? " Other matched " + best.platform_id.toUpperCase() + " series at " + formatInteger(highestGpuCount) + " GPUs ranged down to " + formatDecimal(number(weakest.scaling_efficiency) * 100, 1) + "% efficiency, so scaling is workload- and protocol-dependent."
    : "";
  const supportingRows = exact.filter((row) => row.platform_id === "l40s");
  const supportingGpuCount = max(supportingRows.map((row) => number(row.gpus)));
  const supporting = supportingRows.filter((row) => number(row.gpus) === supportingGpuCount).sort((left, right) => number(right.scaling_efficiency) - number(left.scaling_efficiency))[0];
  const supportingStatement = supporting
    ? " The supporting L40S extension reached " + formatDecimal(number(supporting.speedup), 2) + "× at " + formatInteger(supportingGpuCount) + " GPUs with " + formatDecimal(number(supporting.scaling_efficiency) * 100, 1) + "% efficiency for " + displayWorkload(supporting.workload) + " " + supporting.scaling_type + " scaling."
    : "";
  return {
    id: "scaling",
    researchQuestion: "RQ1",
    tone: efficiency >= 85 ? "positive" : "conditional",
    confidence: "Sufficient repeated evidence",
    title: "Scaling gains are strong, but depend on workload and scaling protocol",
    metric: formatDecimal(number(best.speedup), 2) + "× speedup",
    statement: displayPlatform(best.platform_id) + " " + displayWorkload(best.workload) + " retained " + formatDecimal(efficiency, 1) + "% parallel efficiency at " + formatInteger(highestGpuCount) + " GPUs against the " + formatInteger(number(best.baseline_gpus)) + "-GPU baseline." + supportingStatement + range,
    implication: "Scale-out is valuable for this protocol, but GPU count alone is not a performance guarantee.",
    changed: "GPU count: " + formatInteger(number(best.baseline_gpus)) + " → " + formatInteger(highestGpuCount),
    heldConstant: "Platform, workload, precision, strategy, scaling protocol, throughput unit",
    limitation: best.limitation,
    conditionIds: uniqueStrings([...parseConditionIds(best.source_condition_ids), ...parseConditionIds(weakest?.source_condition_ids), ...parseConditionIds(supporting?.source_condition_ids)]),
    platforms: uniqueStrings([best.platform_id, supporting?.platform_id]),
    workloads: uniqueStrings([best.workload, supporting?.workload]),
    target: "research-scaling",
  };
}

function buildStrategyDecision(rows) {
  const eligible = rows.filter((row) => row.research_question === "RQ2" && row.finding_status === "sufficient" && row.comparability === "exact" && number(row.throughput_tokens_sec) > 0 && number(row.max_memory_used_gb) > 0);
  const preferred = eligible.some((row) => row.platform_id === "h100") ? eligible.filter((row) => row.platform_id === "h100") : eligible;
  const candidates = [];
  Object.values(groupRows(preferred, (row) => [row.platform_id, row.workload, row.precision, row.gpus, row.throughput_unit])).forEach((group) => {
    const baseline = group.find((row) => row.strategy === "ddp");
    if (!baseline) return;
    group.filter((row) => row !== baseline).forEach((row) => {
      const throughputGain = (number(row.throughput_tokens_sec) / number(baseline.throughput_tokens_sec) - 1) * 100;
      const memorySaving = (1 - number(row.max_memory_used_gb) / number(baseline.max_memory_used_gb)) * 100;
      if (throughputGain >= 0 && memorySaving >= 0) candidates.push({ row, baseline, throughputGain, memorySaving, score: throughputGain + memorySaving });
    });
  });
  if (!candidates.length) return null;
  const best = candidates.sort((left, right) => right.score - left.score)[0];
  return {
    id: "strategy",
    researchQuestion: "RQ2",
    tone: "positive",
    confidence: "Matched Pareto comparison",
    title: displayStrategy(best.row.strategy) + " gives the clearest speed–memory improvement over DDP",
    metric: signedPercent(best.throughputGain) + " speed · " + formatDecimal(best.memorySaving, 1) + "% less memory",
    statement: "Within the matched " + displayPlatform(best.row.platform_id) + " " + displayWorkload(best.row.workload) + " " + formatInteger(number(best.row.gpus)) + "-GPU comparison, this strategy improved throughput while reducing maximum GPU memory.",
    implication: "Treat this as a context-specific winner, not a universal strategy ranking.",
    changed: "Distributed strategy: DDP → " + displayStrategy(best.row.strategy),
    heldConstant: "Platform, workload, precision, GPU count, batch semantics, throughput unit",
    limitation: best.row.limitation,
    conditionIds: uniqueStrings([...parseConditionIds(best.baseline.source_condition_ids), ...parseConditionIds(best.row.source_condition_ids)]),
    platforms: [best.row.platform_id],
    workloads: [best.row.workload],
    target: "research-strategy",
  };
}

function buildPrecisionDecision(rows) {
  const completed = rows.filter((row) => row.status === "completed" && row.mode === "training" && row.phase === "Precision study");
  const conditions = groupRows(completed, (row) => [conditionId(row)]);
  const pairs = [];
  Object.entries(conditions).forEach(([treatmentId, treatmentRows]) => {
    const sample = treatmentRows[0];
    if (sample.precision !== "fp8" || !sample.parameter_comparison_control_id) return;
    const controlId = sample.parameter_comparison_control_id + "@" + sample.platform_id;
    const controlRows = conditions[controlId];
    if (!controlRows) return;
    const treatmentThroughput = average(treatmentRows.map(throughput).filter((value) => value > 0));
    const controlThroughput = average(controlRows.map(throughput).filter((value) => value > 0));
    if (!treatmentThroughput || !controlThroughput) return;
    const treatmentMemory = conditionMetricAverage(completed, treatmentId, ["metric_nvidia_smi_memory_used_gb_measured_region", "metric_memory_used_gb", "memory_used_gb"]);
    const controlMemory = conditionMetricAverage(completed, controlId, ["metric_nvidia_smi_memory_used_gb_measured_region", "metric_memory_used_gb", "memory_used_gb"]);
    pairs.push({
      treatmentId,
      controlId,
      platform: sample.platform_id,
      workload: sample.workload,
      throughputGain: (treatmentThroughput / controlThroughput - 1) * 100,
      memorySaving: treatmentMemory !== null && controlMemory ? (1 - treatmentMemory / controlMemory) * 100 : null,
    });
  });
  if (!pairs.length) return null;
  const gains = pairs.map((pair) => pair.throughputGain);
  const savings = pairs.map((pair) => pair.memorySaving).filter((value) => value !== null);
  return {
    id: "precision",
    researchQuestion: "RQ3",
    tone: gains.every((value) => value > 0) ? "positive" : "conditional",
    confidence: "Backend-matched FP8 controls",
    title: "Validated training FP8 improved every backend-matched comparison",
    metric: formatDecimal(Math.min(...gains), 1) + "–" + formatDecimal(Math.max(...gains), 1) + "% faster",
    statement: pairs.length + " Transformer Engine FP8/BF16 pairs cover " + uniqueStrings(pairs.map((pair) => displayWorkload(pair.workload))).join(" and ") + " on " + uniqueStrings(pairs.map((pair) => displayPlatform(pair.platform))).join(" and ") + "." + (savings.length ? " FP8 also used " + formatDecimal(Math.min(...savings), 1) + "–" + formatDecimal(Math.max(...savings), 1) + "% less measured GPU memory." : ""),
    implication: "FP8 is the strongest observed training format where the validated Transformer Engine path is available.",
    changed: "Numeric format: backend-matched BF16 → FP8",
    heldConstant: "Transformer Engine backend, platform, workload, model, batch and sequence length",
    limitation: "Short-run loss is a stability check; it does not establish downstream application quality.",
    conditionIds: uniqueStrings(pairs.flatMap((pair) => [pair.treatmentId, pair.controlId])),
    platforms: uniqueStrings(pairs.map((pair) => pair.platform)),
    workloads: uniqueStrings(pairs.map((pair) => pair.workload)),
    target: "precisionBars",
  };
}

function buildMemoryDecision(rows) {
  const pairs = buildCheckpointingGroups(rows).flatMap((group) => {
    const enabled = group.entries.find((entry) => entry.label === "Checkpointing on");
    const disabled = group.entries.find((entry) => entry.label === "Checkpointing off");
    if (!enabled || !disabled || !disabled.value || !disabled.throughput) return [];
    return [{
      group,
      memorySaving: (1 - enabled.value / disabled.value) * 100,
      throughputCost: (1 - enabled.throughput / disabled.throughput) * 100,
      conditionIds: [disabled.conditionId, enabled.conditionId],
    }];
  });
  if (!pairs.length) return null;
  const memorySavings = pairs.map((pair) => pair.memorySaving);
  const throughputCosts = pairs.map((pair) => pair.throughputCost);
  return {
    id: "memory",
    researchQuestion: "RQ2",
    tone: "conditional",
    confidence: pairs.length + " controlled comparison" + (pairs.length === 1 ? "" : "s"),
    title: "Activation checkpointing buys capacity at a substantial speed cost",
    metric: formatDecimal(Math.min(...memorySavings), 0) + "–" + formatDecimal(Math.max(...memorySavings), 0) + "% less memory",
    statement: "Across the separate matched comparisons, checkpointing reduced peak memory while lowering throughput by " + formatDecimal(Math.min(...throughputCosts), 0) + "–" + formatDecimal(Math.max(...throughputCosts), 0) + "%.",
    implication: "Enable checkpointing when memory capacity is the blocker, not as a default throughput optimisation.",
    changed: "Activation checkpointing: off → on",
    heldConstant: "Platform, workload, micro-batch size, accumulation and throughput unit within each pair",
    limitation: "The displayed ranges summarise separate controlled pairs; their absolute values are not pooled.",
    conditionIds: uniqueStrings(pairs.flatMap((pair) => pair.conditionIds)),
    platforms: uniqueStrings(pairs.map((pair) => pair.group.platform)),
    workloads: uniqueStrings(pairs.map((pair) => pair.group.workload)),
    target: "memory",
  };
}

function buildCommunicationDecision(rows) {
  const matched = [];
  Object.values(groupRows(rows.filter((row) => number(row.gpus) === 2 && number(row.avg_nccl_bandwidth_gbps) > 0), (row) => [row.workload, row.strategy, row.gpus])).forEach((group) => {
    const h100 = group.find((row) => row.platform_id === "h100");
    const l40s = group.find((row) => row.platform_id === "l40s");
    if (!h100 || !l40s) return;
    matched.push({ ratio: number(h100.avg_nccl_bandwidth_gbps) / number(l40s.avg_nccl_bandwidth_gbps), h100, l40s });
  });
  if (!matched.length) return null;
  const ratios = matched.map((entry) => entry.ratio);
  return {
    id: "communication",
    researchQuestion: "RQ4",
    tone: "conditional",
    confidence: "Matched two-GPU microbenchmarks",
    title: "Interconnect capability differs much more than end-to-end throughput alone suggests",
    metric: formatDecimal(Math.min(...ratios), 1) + "–" + formatDecimal(Math.max(...ratios), 1) + "× bandwidth",
    statement: "H100 delivered this multiple of L40S NCCL bandwidth across " + matched.length + " matched two-GPU workload/strategy protocols.",
    implication: "Communication-heavy scale-out is more likely to benefit from H100, but the microbenchmark does not prove the fraction of application time spent communicating.",
    changed: "GPU platform and installed interconnect",
    heldConstant: "Two GPUs, workload, distributed strategy and communication protocol within each pair",
    limitation: "Standalone NCCL measurements establish capability, not in-loop causal attribution.",
    conditionIds: uniqueStrings(matched.flatMap((entry) => [conditionId(entry.h100), conditionId(entry.l40s)])),
    platforms: ["h100", "l40s"],
    workloads: uniqueStrings(matched.map((entry) => entry.h100.workload)),
    target: "scaling",
  };
}

function buildHardwareDecision(rows) {
  const groups = buildHardwareComparisonGroups(rows);
  const training = groups.filter((group) => group.mode === "training").map(hardwareThroughputRatio).filter(Boolean);
  const inference = groups.filter((group) => group.mode === "inference").map(hardwareThroughputRatio).filter(Boolean);
  if (!training.length || !inference.length) return null;
  const trainingRatios = training.map((entry) => entry.ratio);
  const inferenceRatios = inference.map((entry) => entry.ratio);
  return {
    id: "hardware",
    researchQuestion: "RQ6",
    tone: "positive",
    confidence: "Four matched exact protocols",
    title: "H100's advantage is strong for training, not universal for inference",
    metric: formatDecimal(Math.min(...trainingRatios), 2) + "–" + formatDecimal(Math.max(...trainingRatios), 2) + "× training throughput",
    statement: "H100 led both matched training workloads. Inference ranged from " + formatDecimal(Math.min(...inferenceRatios), 2) + "× to " + formatDecimal(Math.max(...inferenceRatios), 2) + "× H100/L40S throughput, including one result where L40S was faster.",
    implication: "Prefer H100 for these training protocols; benchmark the actual inference workload before assuming the same advantage.",
    changed: "GPU platform: L40S ↔ H100",
    heldConstant: "One GPU, exact workload, model, mode, precision, batch and throughput unit",
    limitation: "Energy-to-solution remains pending for the four L40S hardware energy anchors.",
    conditionIds: uniqueStrings(groups.flatMap((group) => group.entries.map((entry) => entry.conditionId))),
    platforms: ["h100", "l40s"],
    workloads: uniqueStrings(groups.map((group) => group.workload)),
    target: "hardware",
  };
}

function hardwareThroughputRatio(group) {
  const h100 = group.entries.find((entry) => entry.key === "h100");
  const l40s = group.entries.find((entry) => entry.key === "l40s");
  return h100 && l40s && l40s.value ? { ratio: h100.value / l40s.value, h100, l40s } : null;
}

function buildEnergyDecision(rows) {
  const eligible = rows.filter((row) => row.status === "completed" && row.energy_scope === "measured_region" && row.energy_measurement_status === "sufficient" && number(row.energy_joules) > 0);
  if (!eligible.length) return null;
  const conditions = uniqueConditionIds(eligible);
  const pendingHardware = ["E123@l40s", "E124@l40s", "E127@l40s", "E128@l40s"].filter((id) => rows.some((row) => conditionId(row) === id && !number(row.energy_joules)));
  return {
    id: "energy",
    researchQuestion: "RQ5",
    tone: pendingHardware.length ? "limitation" : "positive",
    confidence: pendingHardware.length ? "Partial cross-platform coverage" : "Measured-region evidence",
    title: pendingHardware.length ? "Energy analysis is broad, but the hardware comparison is not yet complete" : "Measured-region energy evidence is available",
    metric: formatInteger(conditions.length) + " measured conditions",
    statement: formatInteger(eligible.length) + " completed trials have sufficient measured-region energy." + (pendingHardware.length ? " The four original L40S hardware-comparison rows remain excluded from energy-to-solution conclusions until E235–E238 run." : ""),
    implication: pendingHardware.length ? "Continue analysing scoped energy studies, but defer the final H100-versus-L40S energy conclusion." : "Energy comparisons may proceed only within matched protocols.",
    changed: "Depends on the selected controlled energy study",
    heldConstant: "Only schema-v3, sufficient, measured-region records are admitted",
    limitation: "Legacy full-window telemetry is retained for provenance but excluded from canonical joules and tokens/kWh.",
    conditionIds: [],
    platforms: uniqueValues(eligible, "platform_id"),
    workloads: uniqueValues(eligible, "workload"),
    target: "energyCoverage",
  };
}

function uniqueStrings(values) {
  return [...new Set(values.filter(Boolean))];
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
  buildResearchQuestionOverview(rows).forEach((row) => {
    const card = document.createElement("article");
    card.className = "research-question-card research-status-" + row.status;
    const completion = row.requiredTrials ? Math.min(100, row.completedTrials / row.requiredTrials * 100) : 0;
    const energyNote = row.researchQuestion === "RQ5" ? '<span class="rq-energy-note">' + formatInteger(row.energyTrials) + " measured-region energy trials</span>" : "";
    card.innerHTML = '<div class="research-card-heading"><div><span class="rq-index">' + escapeHtml(row.researchQuestion) + '</span><strong>' + escapeHtml(row.objective) + '</strong></div><span class="evidence-badge evidence-' + escapeHtml(row.status) + '">' + escapeHtml(row.statusLabel) + "</span></div>" +
      '<p class="research-question-text">' + escapeHtml(row.question) + "</p>" +
      '<div class="rq-platforms">' + row.platforms.map((platform) => "<span>" + escapeHtml(platform) + "</span>").join("") + "</div>" +
      '<div class="rq-progress"><div><span>Trial coverage</span><strong>' + formatInteger(row.completedTrials) + "/" + formatInteger(row.requiredTrials) + '</strong></div><div class="rq-progress-track"><i style="width:' + completion + '%"></i></div></div>' +
      '<div class="rq-evidence-line"><span>' + formatInteger(row.conditions) + " direct conditions</span>" + energyNote + "</div>" +
      '<p class="research-note">' + escapeHtml(row.limitation) + "</p>" +
      '<a class="rq-link" href="#' + escapeHtml(row.target) + '">Inspect the evidence <span aria-hidden="true">→</span></a>';
    container.append(card);
  });
}

function researchCoverageLabel(rows) {
  const questions = buildResearchQuestionOverview(rows);
  const ready = questions.filter((row) => row.status === "ready").length;
  return ready + " of " + questions.length + " research questions answer-ready";
}

function buildRq1ScalingSeries(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.speedup) > 0 && number(row.gpus) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.precision || "unspecified", row.strategy || "unspecified", row.scaling_type || "unspecified", row.throughput_unit || "unit_not_recorded"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, precision, strategy, scalingType, unit] = key.split("\u0000");
    const points = group.map((row) => ({
      x: number(row.gpus),
      y: number(row.speedup),
      label: row.gpus + " GPUs: " + formatDecimal(number(row.speedup), 2) + "× speedup; " + formatDecimal(number(row.scaling_efficiency) * 100, 1) + "% efficiency; " + row.source_condition_ids,
      conditionIds: parseConditionIds(row.source_condition_ids),
    }));
    const baselineGpus = number(group[0].baseline_gpus);
    if (baselineGpus > 0 && !points.some((point) => point.x === baselineGpus)) {
      points.push({
        x: baselineGpus,
        y: 1,
        label: baselineGpus + " GPU baseline: 1.00× speedup; 100.0% reference efficiency",
        conditionIds: parseConditionIds(group[0].source_condition_ids).slice(0, 1),
      });
    }
    return {
      title: platform + " · " + comparability + " · " + workload,
      subtitle: scalingType + " scaling; " + precision + "; " + strategy + "; " + unit,
      platform, comparability, workload, scalingType, unit,
      comparison: comparisonContext(group, ["GPU count"], ["platform_id", "comparability", "workload", "precision", "strategy", "scaling_type", "throughput_unit"]),
      points: points.sort((left, right) => left.x - right.x),
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
  const matchedProtocols = buildBroadHardwareComparisonGroups(filteredTrialRowsWithEvidence(), state.results).length;
  document.getElementById("kpiExperiments").textContent = formatInteger(conditions);
  document.getElementById("kpiCompleted").textContent = formatInteger(completed);
  document.getElementById("kpiGpuHours").textContent = formatDecimal(gpuHours, 2);
  document.getElementById("kpiMatchedProtocols").textContent = formatInteger(matchedProtocols);
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
  safeRender();
}

function resetFilters() {
  filterDefinitions.forEach(([id]) => { document.getElementById(id).value = ""; });
  document.getElementById("evidenceFilter").value = "";
  document.getElementById("metricFilter").value = "";
  document.getElementById("strategyMetricFilter").value = "throughput";
  document.getElementById("scalingTypeFilter").value = "";
  safeRender();
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

function renderStudyMap(rows) {
  const container = document.getElementById("studyMap");
  const entries = buildStudyCoverageEntries(rows);
  container.replaceChildren();
  if (!entries.length) {
    container.append(emptyState("No completed throughput evidence matches the active filters."));
  } else {
    entries.forEach((entry) => {
      const card = document.createElement("article");
      card.className = "study-card";
      const badges = entry.researchQuestions.length
        ? entry.researchQuestions.map((question) => '<span class="study-rq">' + escapeHtml(question) + "</span>").join("")
        : '<span class="study-rq study-rq-muted">Supporting evidence</span>';
      card.innerHTML =
        '<div class="study-card-heading"><h3>' + escapeHtml(entry.label) + '</h3><div class="study-rqs">' + badges + "</div></div>" +
        '<div class="study-variable"><span>Variable tested</span><strong>' + escapeHtml(entry.changing) + "</strong></div>" +
        '<dl class="study-coverage"><div><dt>Conditions</dt><dd>' + formatInteger(entry.conditionIds.length) + '</dd></div><div><dt>Trials</dt><dd>' + formatInteger(entry.trials) + "</dd></div></dl>" +
        '<p class="study-scope">' + escapeHtml(entry.platforms.join(" + ") || "No platform") + " · " + escapeHtml(entry.workloads.join(" + ") || "No workload") + "</p>" +
        '<a class="study-link" href="#' + escapeHtml(entry.target) + '">Open controlled view <span aria-hidden="true">→</span></a>';
      container.append(card);
    });
  }
  const conditionCount = sum(entries.map((entry) => entry.conditionIds.length));
  document.getElementById("studyMapStatus").textContent = entries.length
    ? entries.length + " study families · " + formatInteger(conditionCount) + " conditions"
    : "No study evidence";
  markPanel("studyMap", entries.length > 0, buildStudyCoverageEntries(state.results).length > 0, "Controlled study map", "experiments.csv", true);
}

function buildStudyCoverageEntries(rows) {
  const eligible = rows.filter((row) => row.status === "completed" && throughput(row) > 0);
  const groups = groupRows(eligible, (row) => {
    const study = studyFamily(row.phase);
    return [study.label, study.target, study.changing];
  });
  return Object.entries(groups).map(([key, group]) => {
    const [label, target, changing] = key.split("\u0000");
    return {
      label,
      target,
      changing,
      conditionIds: uniqueConditionIds(group),
      trials: group.length,
      platforms: [...new Set(group.map((row) => row.platform_id).filter(Boolean))].sort(),
      workloads: [...new Set(group.map((row) => row.workload).filter(Boolean))].sort(),
      researchQuestions: [...new Set(group.flatMap(researchQuestionsForRow))].sort(),
    };
  }).sort((left, right) => left.label.localeCompare(right.label));
}

function studyFamily(phase) {
  const value = String(phase || "Unassigned");
  if (value === "Strong scaling" || value === "Weak scaling") {
    return { label: "Scaling efficiency", target: "scaling", changing: "GPU count" };
  }
  if (value === "Distributed training strategy") {
    return { label: "Distributed strategies", target: "performance", changing: "DDP, FSDP, or ZeRO strategy" };
  }
  if (value === "Memory optimisation") {
    return { label: "Memory optimisation", target: "memory", changing: "Micro-batch, checkpointing, or accumulation" };
  }
  if (value === "Precision study" || value === "Inference precision/quantization") {
    return { label: "Precision and quantisation", target: "performance", changing: "Numeric precision" };
  }
  if (value === "Communication analysis" || value === "Data movement" || value === "Workload characterisation") {
    return { label: "Bottleneck analysis", target: "scaling", changing: "Communication or data-movement setting" };
  }
  if (value.startsWith("Inference")) {
    return { label: "Inference behaviour", target: "inference", changing: "Framework, request shape, or parallelism" };
  }
  if (value === "Hardware comparison") {
    return { label: "Hardware comparison", target: "research", changing: "GPU platform" };
  }
  if (value === "Telemetry revalidation") {
    return { label: "Useful-work efficiency", target: "overview", changing: "Measured workload condition" };
  }
  return { label: "Baseline and validation", target: "evidence", changing: "Workload or validation condition" };
}

function buildPhaseDistributionEntries(rows) {
  const eligible = rows.filter((row) => row.status === "completed" && throughput(row) > 0);
  const byCondition = groupRows(eligible, (row) => [conditionId(row), row.throughput_unit || "unit_not_recorded"]);
  const conditions = Object.values(byCondition).map((group) => {
    const first = group[0];
    return {
      platform: first.platform_id || "unlabelled",
      comparability: first.comparability || "unspecified",
      phase: first.phase || "Unassigned",
      workload: first.workload || "Unassigned",
      unit: first.throughput_unit || "unit_not_recorded",
      conditionId: conditionId(first),
      value: average(group.map(throughput).filter((value) => value > 0)),
      trials: group.length,
    };
  });
  const groups = groupRows(conditions, (row) => [row.platform, row.comparability, row.phase, row.workload, row.unit]);
  return Object.entries(groups).map(([key, group]) => {
    const [platform, comparability, phase, workload, unit] = key.split("\u0000");
    const values = group.map((row) => row.value).filter((value) => value > 0).sort((left, right) => left - right);
    return {
      label: phase,
      platform,
      comparability,
      phase,
      workload,
      unit,
      conditionIds: group.map((row) => row.conditionId).filter(Boolean),
      conditions: group.length,
      trials: sum(group.map((row) => row.trials)),
      median: median(values),
      minimum: values[0] || 0,
      maximum: values[values.length - 1] || 0,
    };
  }).filter((entry) => entry.median > 0).sort((left, right) => {
    const leftKey = [left.platform, left.comparability, left.workload, left.unit, left.phase].join("|");
    const rightKey = [right.platform, right.comparability, right.workload, right.unit, right.phase].join("|");
    return leftKey.localeCompare(rightKey);
  });
}

function renderEnergy(rows) {
  const entries = buildEnergyPowerEntries(rows);
  const efficiencyGroups = buildEnergyEfficiencyGroups(rows);
  const scalingSeries = buildEnergyPowerScalingSeries(rows);
  const solutionGroups = buildEnergyToSolutionGroups(rows);
  renderEnergyCoverage(entries, efficiencyGroups, scalingSeries, solutionGroups);
  renderEnergyEfficiencyCharts("energyEfficiencyCharts", efficiencyGroups);
  renderEnergyPowerScalingCharts("energyScalingCharts", scalingSeries);
  renderEnergyToSolutionCharts("energySolutionCharts", solutionGroups);
  renderSeparatedBars("energyBars", entries, {
    empty: "No measured-region energy evidence is available. Legacy full-window estimates are excluded.",
    barClass: "energy",
    value: (entry) => formatDecimal(entry.value, 0) + " J · " + formatDecimal(entry.power, 1) + " W aggregate · " + formatDecimal(entry.powerPerGpu, 1) + " W/GPU",
    changing: (group) => varyingLabels(group, ["platform", "comparability", "experiment", "workload", "mode"]),
    constants: ["platform", "comparability", "workload", "mode"],
  });
  document.getElementById("energyStatus").textContent = entries.length ? entries.length + " measured conditions · " + sum(efficiencyGroups.map((group) => group.points.length)) + " efficiency observations" : "No energy evidence";
  document.getElementById("energyEvidenceStatus").textContent = entries.length ? entries.length + " conditions · click to expand" : "No conditions";
  markPanel("energyEfficiencyCharts", entries.length > 0, buildEnergyPowerEntries(state.results).length > 0, "Energy efficiency and power", "experiments.csv");
}

function renderEnergyCoverage(entries, efficiencyGroups, scalingSeries, solutionGroups) {
  const container = document.getElementById("energyCoverage");
  container.replaceChildren();
  if (!entries.length) return;
  const exactPlatforms = [...new Set(entries.filter((entry) => entry.comparability === "exact").map((entry) => entry.platform))];
  const gpuCounts = [...new Set(entries.map((entry) => entry.gpus))].sort((left, right) => left - right);
  const summary = document.createElement("div");
  summary.className = "energy-coverage-summary";
  summary.innerHTML = '<div><span>Measured conditions</span><strong>' + entries.length + '</strong></div><div><span>Controlled power series</span><strong>' + scalingSeries.length + '</strong></div><div><span>Fixed-work comparisons</span><strong>' + solutionGroups.length + '</strong></div><div><span>GPU counts represented</span><strong>' + escapeHtml(gpuCounts.join(", ")) + '</strong></div>';
  container.append(summary);
  const note = document.createElement("p");
  note.className = exactPlatforms.length > 1 ? "energy-scope-note" : "energy-scope-note is-limited";
  note.textContent = exactPlatforms.length > 1
    ? "Exact measured-region evidence is available for " + exactPlatforms.join(" and ") + "; cross-platform comparisons still require matching workload and protocol."
    : "Cross-platform energy comparison is pending: exact measured-region evidence currently covers " + (exactPlatforms.join(", ") || "no platform") + ". Platform-specific rows remain descriptive only.";
  container.append(note);
  if (!efficiencyGroups.length) {
    const missing = document.createElement("p");
    missing.className = "energy-scope-note is-limited";
    missing.textContent = "Efficiency requires both measured power and throughput; no eligible paired observations match the active filters.";
    container.append(missing);
  }
}

function renderEnergyEfficiencyCharts(containerId, groups) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState("No paired throughput and measured-power evidence matches the active filters."));
    return;
  }
  groups.forEach((group) => container.append(energyEfficiencyChartCard(group)));
}

function energyEfficiencyChartCard(group) {
  const card = chartCard(group.title, group.subtitle + " · " + group.points.length + " conditions", group.comparison);
  const width = 520;
  const height = 250;
  const left = 66;
  const right = 20;
  const top = 18;
  const bottom = 42;
  const xDomain = paddedDomain(group.points.map((point) => point.x), true);
  const yDomain = paddedDomain(group.points.flatMap((point) => [point.y, point.efficiencyMin, point.efficiencyMax]), true);
  const x = linearScale(xDomain[0], xDomain[1], left, width - right);
  const y = linearScale(yDomain[0], yDomain[1], height - bottom, top);
  const xTicks = [xDomain[0], (xDomain[0] + xDomain[1]) / 2, xDomain[1]];
  const yTicks = [yDomain[0], (yDomain[0] + yDomain[1]) / 2, yDomain[1]];
  const topPower = max(group.points.map((point) => point.power)) || 1;
  const grid = xTicks.map((value) => '<line class="svg-grid" x1="' + x(value) + '" y1="' + top + '" x2="' + x(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label" x="' + x(value) + '" y="' + (height - 24) + '" text-anchor="middle">' + formatCompact(value) + '</text>').join("") + yTicks.map((value) => '<line class="svg-grid" x1="' + left + '" y1="' + y(value) + '" x2="' + (width - right) + '" y2="' + y(value) + '"/><text class="svg-label" x="' + (left - 6) + '" y="' + (y(value) + 3) + '" text-anchor="end">' + formatCompact(value) + '</text>').join("");
  const points = group.points.map((point) => {
    const radius = 4 + Math.sqrt(point.power / topPower) * 6;
    const color = colors[(Math.max(1, point.gpus) - 1) % colors.length];
    const title = point.label + "; " + formatCompact(point.x) + " " + displayName(group.throughputUnit) + "; " + formatCompact(point.y) + " tokens/kWh; " + formatDecimal(point.power, 1) + " W aggregate; " + point.trials + " trials";
    const uncertainty = point.trials > 1 && point.efficiencyMax > point.efficiencyMin
      ? '<line class="svg-whisker" x1="' + x(point.x) + '" y1="' + y(point.efficiencyMin) + '" x2="' + x(point.x) + '" y2="' + y(point.efficiencyMax) + '"/><line class="svg-whisker" x1="' + (x(point.x) - 3) + '" y1="' + y(point.efficiencyMin) + '" x2="' + (x(point.x) + 3) + '" y2="' + y(point.efficiencyMin) + '"/><line class="svg-whisker" x1="' + (x(point.x) - 3) + '" y1="' + y(point.efficiencyMax) + '" x2="' + (x(point.x) + 3) + '" y2="' + y(point.efficiencyMax) + '"/>'
      : '';
    return uncertainty + '<circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="' + color + '" cx="' + x(point.x) + '" cy="' + y(point.y) + '" r="' + radius + '"><title>' + escapeSvg(title) + '</title></circle>';
  }).join("");
  const labels = '<text class="svg-label axis-title" x="' + width / 2 + '" y="' + (height - 4) + '" text-anchor="middle">throughput (' + escapeSvg(displayName(group.throughputUnit)) + ')</text><text class="svg-label axis-title" transform="translate(12 ' + height / 2 + ') rotate(-90)" text-anchor="middle">tokens per kWh</text>';
  card.append(svgElement(width, height, grid + points + labels));
  card.append(energyLegend(group.points));
  return card;
}

function energyLegend(points) {
  const legend = document.createElement("div");
  legend.className = "energy-legend";
  const counts = [...new Set(points.map((point) => point.gpus))].sort((left, right) => left - right);
  legend.innerHTML = counts.map((gpus) => '<span><i style="--legend-color:' + colors[(Math.max(1, gpus) - 1) % colors.length] + '"></i>' + gpus + ' GPU' + (gpus === 1 ? '' : 's') + '</span>').join("") + '<span class="energy-size-key">circle size = aggregate power</span>';
  return legend;
}

function renderEnergyPowerScalingCharts(containerId, series) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!series.length) {
    container.append(emptyState("No multi-GPU measured-power series matches the active filters."));
    return;
  }
  series.forEach((group) => container.append(energyPowerScalingChartCard(group)));
}

function energyPowerScalingChartCard(group) {
  const card = chartCard(group.title, group.subtitle, group.comparison);
  const width = 520;
  const height = 245;
  const left = 62;
  const right = 20;
  const top = 18;
  const bottom = 40;
  const xDomain = paddedDomain(group.points.map((point) => point.x), false);
  const yDomain = paddedDomain(group.points.flatMap((point) => [point.aggregatePower, point.perGpuPower]), true);
  const x = linearScale(xDomain[0], xDomain[1], left, width - right);
  const y = linearScale(yDomain[0], yDomain[1], height - bottom, top);
  const xTicks = pointAxisTicks(group.points.map((point) => point.x), xDomain);
  const yTicks = [yDomain[0], (yDomain[0] + yDomain[1]) / 2, yDomain[1]];
  const grid = xTicks.map((value) => '<line class="svg-grid" x1="' + x(value) + '" y1="' + top + '" x2="' + x(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label" x="' + x(value) + '" y="' + (height - 22) + '" text-anchor="middle">' + formatCompact(value) + '</text>').join("") + yTicks.map((value) => '<line class="svg-grid" x1="' + left + '" y1="' + y(value) + '" x2="' + (width - right) + '" y2="' + y(value) + '"/><text class="svg-label" x="' + (left - 6) + '" y="' + (y(value) + 3) + '" text-anchor="end">' + formatCompact(value) + '</text>').join("");
  const aggregateLine = group.points.map((point, index) => (index ? "L" : "M") + x(point.x) + "," + y(point.aggregatePower)).join(" ");
  const perGpuLine = group.points.map((point, index) => (index ? "L" : "M") + x(point.x) + "," + y(point.perGpuPower)).join(" ");
  const aggregatePoints = group.points.map((point) => '<circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="#b45309" cx="' + x(point.x) + '" cy="' + y(point.aggregatePower) + '" r="5"><title>' + escapeSvg(point.label + "; aggregate " + formatDecimal(point.aggregatePower, 1) + " W; " + formatCompact(point.throughput) + " tokens/s") + '</title></circle>').join("");
  const perGpuPoints = group.points.map((point) => '<circle class="svg-point" ' + conditionTargetAttributes(point) + ' fill="#0f766e" cx="' + x(point.x) + '" cy="' + y(point.perGpuPower) + '" r="4"><title>' + escapeSvg(point.label + "; " + formatDecimal(point.perGpuPower, 1) + " W per GPU") + '</title></circle>').join("");
  const labels = '<text class="svg-label axis-title" x="' + width / 2 + '" y="' + (height - 4) + '" text-anchor="middle">GPU count</text><text class="svg-label axis-title" transform="translate(12 ' + height / 2 + ') rotate(-90)" text-anchor="middle">power (W)</text>';
  card.append(svgElement(width, height, grid + '<path class="svg-line" stroke="#b45309" d="' + aggregateLine + '"/><path class="svg-line energy-per-gpu-line" stroke="#0f766e" d="' + perGpuLine + '"/>' + aggregatePoints + perGpuPoints + labels));
  const legend = document.createElement("div");
  legend.className = "energy-legend";
  legend.innerHTML = '<span><i style="--legend-color:#b45309"></i>aggregate power</span><span><i style="--legend-color:#0f766e"></i>power per GPU</span>';
  card.append(legend);
  return card;
}

function renderEnergyToSolutionCharts(containerId, groups) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState("No fixed-work distributed-strategy comparison matches the active filters."));
    return;
  }
  groups.forEach((group) => container.append(energyToSolutionChartCard(group)));
}

function energyToSolutionChartCard(group) {
  const card = chartCard(group.title, group.subtitle, group.comparison);
  const width = 520;
  const rowHeight = 35;
  const height = 54 + group.entries.length * rowHeight;
  const left = 168;
  const right = 82;
  const top = 18;
  const scaleMaximum = max(group.entries.map((entry) => entry.energyMax)) || 1;
  const scale = linearScale(0, scaleMaximum, left, width - right);
  const bars = group.entries.map((entry, index) => {
    const y = top + index * rowHeight;
    const label = entry.strategyLabel.replaceAll("_", " ");
    const title = entry.experiment + "; " + formatDecimal(entry.value, 0) + " J; " + formatCompact(entry.throughput) + " " + displayName(entry.throughputUnit) + "; measured " + formatDecimal(entry.duration, 2) + " s; range " + formatDecimal(entry.energyMin, 0) + "–" + formatDecimal(entry.energyMax, 0) + " J";
    const whiskerY = y + 9;
    const whisker = entry.trials > 1 && entry.energyMax > entry.energyMin
      ? '<line class="svg-whisker energy-bar-whisker" x1="' + scale(entry.energyMin) + '" y1="' + whiskerY + '" x2="' + scale(entry.energyMax) + '" y2="' + whiskerY + '"/><line class="svg-whisker energy-bar-whisker" x1="' + scale(entry.energyMin) + '" y1="' + (whiskerY - 4) + '" x2="' + scale(entry.energyMin) + '" y2="' + (whiskerY + 4) + '"/><line class="svg-whisker energy-bar-whisker" x1="' + scale(entry.energyMax) + '" y1="' + (whiskerY - 4) + '" x2="' + scale(entry.energyMax) + '" y2="' + (whiskerY + 4) + '"/>'
      : '';
    return '<text class="svg-label energy-strategy-label" x="' + (left - 8) + '" y="' + (y + 13) + '" text-anchor="end">' + escapeSvg(label) + '</text><rect class="svg-energy-bar" ' + conditionTargetAttributes(entry) + ' x="' + left + '" y="' + y + '" width="' + Math.max(3, scale(entry.value) - left) + '" height="18" rx="3"><title>' + escapeSvg(title) + '</title></rect>' + whisker + '<text class="svg-value" x="' + (width - right + 7) + '" y="' + (y + 13) + '">' + formatCompact(entry.value) + ' J</text>';
  }).join("");
  const axis = '<line class="svg-axis" x1="' + left + '" y1="' + (height - 27) + '" x2="' + (width - right) + '" y2="' + (height - 27) + '"/><text class="svg-label axis-title" x="' + ((left + width - right) / 2) + '" y="' + (height - 5) + '" text-anchor="middle">measured-region energy (J) · lower is better</text>';
  card.append(svgElement(width, height, bars + axis));
  return card;
}

function buildEnergyPowerEntries(rows) {
  const eligible = rows.filter((row) => (!row.status || row.status === "completed") && row.energy_scope === "measured_region" && number(row.energy_joules) > 0);
  const groups = groupRows(eligible, (row) => [conditionId(row) || [row.platform_id, row.comparability, row.experiment_id, row.workload, row.mode].join("|")]);
  return Object.values(groups).map((group) => {
    const first = group[0];
    const powerValues = group.map(energyPower).filter((value) => value > 0);
    const throughputValues = group.map(throughput).filter((value) => value > 0);
    const energyValues = group.map((row) => number(row.energy_joules)).filter((value) => value > 0);
    const durationValues = group.map((row) => number(row.measurement_duration_seconds || row.metric_nvidia_smi_measurement_duration_seconds)).filter((value) => value > 0);
    const efficiencyValues = group.map((row) => {
      const power = energyPower(row);
      const rate = throughput(row);
      return power > 0 && rate > 0 ? rate / power * 3_600_000 : 0;
    }).filter((value) => value > 0);
    const gpus = Math.max(1, number(first.gpus || first.metric_gpu_count));
    const strategyDetail = first.parameter_strategy_detail || first.parameter_distributed_strategy || "";
    const strategy = first.strategy || "none";
    const strategyLabel = strategyDetail && strategyDetail !== strategy ? strategy + " · " + strategyDetail : strategy;
    return {
      label: first.platform_id + " · " + first.comparability + " · " + first.experiment_id + " · " + first.workload,
      detail: first.mode,
      platform: first.platform_id || "unlabelled",
      platform_id: first.platform_id || "unlabelled",
      comparability: first.comparability || "unspecified",
      experiment: first.experiment_id || "unknown",
      experiment_id: first.experiment_id || "unknown",
      workload: first.workload || "unassigned",
      mode: first.mode || "unassigned",
      precision: first.precision || "unspecified",
      phase: first.phase || "unassigned",
      scalingType: first.scaling_type || "",
      scaling_type: first.scaling_type || "",
      strategy,
      strategyDetail,
      strategyLabel,
      gpus,
      throughputUnit: first.throughput_unit || "",
      throughput_unit: first.throughput_unit || "",
      globalBatchSize: first.global_batch_size || "",
      global_batch_size: first.global_batch_size || "",
      perGpuBatchSize: first.per_gpu_batch_size || "",
      per_gpu_batch_size: first.per_gpu_batch_size || "",
      sequenceLength: first.metric_sequence_length || first.parameter_sequence_length || "",
      sequence_length: first.metric_sequence_length || first.parameter_sequence_length || "",
      steps: first.metric_steps || first.measured_iterations || "",
      conditionId: conditionId(first),
      unit: "joules",
      trials: group.length,
      value: average(energyValues),
      energyMin: Math.min(...energyValues),
      energyMax: Math.max(...energyValues),
      power: average(powerValues),
      powerMin: powerValues.length ? Math.min(...powerValues) : 0,
      powerMax: powerValues.length ? Math.max(...powerValues) : 0,
      powerPerGpu: average(powerValues) / gpus,
      throughput: average(throughputValues),
      duration: average(durationValues),
      tokensPerKwh: average(efficiencyValues),
      tokensPerKwhMin: efficiencyValues.length ? Math.min(...efficiencyValues) : 0,
      tokensPerKwhMax: efficiencyValues.length ? Math.max(...efficiencyValues) : 0,
    };
  }).sort((left, right) => right.value - left.value);
}

function energyPower(row) {
  return number(row.metric_nvidia_smi_power_draw_watts_measured_region || row.metric_power_draw_watts || row.power_draw_watts);
}

function buildEnergyEfficiencyGroups(rows) {
  const groups = groupRows(buildEnergyPowerEntries(rows).filter((entry) => entry.throughput > 0 && entry.power > 0 && entry.throughputUnit), (entry) => [entry.platform, entry.comparability, entry.workload, entry.mode, entry.throughputUnit]);
  return Object.values(groups).map((group) => ({
    title: group[0].platform + " · " + group[0].workload + " · " + group[0].mode,
    subtitle: group[0].comparability + " · " + displayName(group[0].throughputUnit),
    platform: group[0].platform,
    comparability: group[0].comparability,
    workload: group[0].workload,
    mode: group[0].mode,
    throughputUnit: group[0].throughputUnit,
    comparison: comparisonContext(group, varyingLabels(group, ["experiment", "gpus", "strategyLabel", "phase"]), ["platform", "comparability", "workload", "mode", "precision", "throughputUnit"]),
    points: group.map((entry) => ({
      x: entry.throughput,
      y: entry.tokensPerKwh,
      power: entry.power,
      gpus: entry.gpus,
      label: entry.experiment + " · " + entry.strategyLabel,
      conditionId: entry.conditionId,
      energyMin: entry.energyMin,
      energyMax: entry.energyMax,
      efficiencyMin: entry.tokensPerKwhMin,
      efficiencyMax: entry.tokensPerKwhMax,
      trials: entry.trials,
    })),
  })).sort((left, right) => right.points.length - left.points.length || left.title.localeCompare(right.title));
}

function buildEnergyPowerScalingSeries(rows) {
  const groups = groupRows(buildEnergyPowerEntries(rows), (entry) => [entry.platform, entry.comparability, entry.workload, entry.mode, entry.precision, entry.strategyLabel, entry.throughputUnit, entry.phase, entry.scalingType]);
  return Object.values(groups).filter((group) => new Set(group.map((entry) => entry.gpus)).size > 1).map((group) => {
    const first = group[0];
    return {
      title: first.platform + " · " + first.workload + " · " + first.phase,
      subtitle: first.comparability + " · " + first.strategyLabel + " · aggregate and per-GPU power",
      comparison: comparisonContext(group, ["GPU count"], ["platform", "comparability", "workload", "mode", "precision", "strategyLabel", "throughputUnit", "phase", "scalingType", "global_batch_size", "per_gpu_batch_size"]),
      points: group.map((entry) => ({
        x: entry.gpus,
        aggregatePower: entry.power,
        perGpuPower: entry.powerPerGpu,
        energy: entry.value,
        throughput: entry.throughput,
        label: entry.experiment,
        conditionId: entry.conditionId,
      })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function buildEnergyToSolutionGroups(rows) {
  const strategyEntries = buildEnergyPowerEntries(rows).filter((entry) => entry.phase === "Distributed training strategy" && entry.throughput > 0);
  const groups = groupRows(strategyEntries, (entry) => [entry.platform, entry.comparability, entry.workload, entry.mode, entry.gpus, entry.precision, entry.globalBatchSize, entry.sequenceLength, entry.steps, entry.throughputUnit, entry.phase]);
  return Object.values(groups).filter((group) => new Set(group.map((entry) => entry.strategyLabel)).size > 1).map((group) => {
    const first = group[0];
    return {
      title: first.platform + " · " + first.workload + " · " + first.gpus + " GPUs",
      subtitle: first.comparability + " · fixed-work strategy comparison",
      comparison: comparisonContext(group, ["Distributed strategy"], ["platform", "comparability", "workload", "mode", "gpus", "precision", "global_batch_size", "sequence_length", "steps", "throughputUnit"]),
      entries: [...group].sort((left, right) => left.value - right.value),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
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

function renderUncertainty() {
  const globalPhase = document.getElementById("phaseFilter").value;
  const studyControl = document.getElementById("trainingStudyFilter");
  const selectedStudy = globalPhase || studyControl.value;
  studyControl.disabled = Boolean(globalPhase);
  studyControl.title = globalPhase ? "The dashboard Phase filter is controlling this study." : "Choose a training study.";
  const trainingRows = filteredTrialRowsWithEvidence().filter((row) => row.mode === "training" && (!selectedStudy || row.phase === selectedStudy));
  const groups = buildUncertaintyGroups(trainingRows, state.results);
  renderErrorBarCharts("uncertaintyCharts", groups, "No completed repeated training runs match this study and the active dashboard filters.");
  renderTrainingStudySummary(groups, selectedStudy, "");
  const configurations = sum(groups.map((group) => group.points.length));
  document.getElementById("uncertaintyStatus").textContent = groups.length + " comparable group" + (groups.length === 1 ? "" : "s") + " · " + configurations + " configuration" + (configurations === 1 ? "" : "s");
  markPanel("uncertaintyCharts", groups.length > 0, buildUncertaintyGroups(state.trialSummary.filter((row) => row.mode === "training"), state.results).length > 0, "Training repeatability", "trial_summary.csv");
}

function filteredTrialRowsWithEvidence() {
  const conditionIds = new Set(filteredResults().map(conditionId).filter(Boolean));
  const evidenceStatus = document.getElementById("evidenceFilter").value;
  return state.trialSummary.filter((row) => conditionIds.has(conditionId(row)) && (!evidenceStatus || row.evidence_status === evidenceStatus));
}

function buildUncertaintyGroups(rows, evidenceRows = []) {
  const metadata = conditionMetadata(evidenceRows);
  const enriched = rows
    .filter((row) => number(row.throughput_tokens_mean) > 0 && number(row.completed_trials) > 0)
    .map((row) => ({ ...(metadata.get(conditionId(row)) || {}), ...row }));
  const grouped = groupRows(enriched, (row) => [
    row.phase || "Unclassified study",
    row.platform_id || "unlabelled",
    row.comparability || "unspecified",
    row.workload || "unassigned",
    row.throughput_unit || "unit_not_recorded",
    trainingComparisonSubgroup(row),
  ]);
  return Object.entries(grouped).map(([key, group]) => {
    const [phase, platform, comparability, workload, unit] = key.split("\u0000");
    const variableKeys = trainingVariableKeys(phase, group);
    const constants = trainingConstantFields(phase, group);
    return {
      title: displayPlatform(platform) + " · " + displayComparability(comparability) + " · " + displayWorkload(workload),
      subtitle: humanizeThroughputUnit(unit),
      phase, platform, comparability, workload, unit,
      comparison: comparisonContext(
        group,
        variableKeys.length ? variableKeys.map(trainingVariableLabel) : ["Repeated runs of one configuration"],
        constants,
      ),
      points: group.map((row) => ({
        label: trainingConfigurationLabel(row, variableKeys),
        meta: trainingPointMeta(row),
        value: number(row.throughput_tokens_mean),
        low: number(row.throughput_tokens_ci95_low) || number(row.throughput_tokens_mean),
        high: number(row.throughput_tokens_ci95_high) || number(row.throughput_tokens_mean),
        trials: number(row.completed_trials),
        cvPercent: number(row.throughput_tokens_mean) > 0 ? number(row.throughput_tokens_std) / number(row.throughput_tokens_mean) * 100 : 0,
        hasInterval: number(row.completed_trials) > 1 && number(row.throughput_tokens_ci95_high) > number(row.throughput_tokens_ci95_low),
        conditionId: conditionId(row),
      })).sort((left, right) => right.value - left.value).slice(0, 10),
    };
  }).sort((left, right) => (left.platform + left.workload + left.title).localeCompare(right.platform + right.workload + right.title));
}

function conditionMetadata(rows) {
  const metadata = new Map();
  rows.filter((row) => row.status === "completed").forEach((row) => {
    const id = conditionId(row);
    if (id && !metadata.has(id)) metadata.set(id, row);
  });
  return metadata;
}

function trainingComparisonSubgroup(row) {
  if (row.phase === "Distributed training strategy") return "gpus=" + (row.gpus || "unspecified");
  if (row.phase === "Memory optimisation") {
    return [row.parameter_memory_study_type || "memory", row.gpus || "unspecified", row.strategy || "unspecified"].join("|");
  }
  return "study";
}

function trainingVariableKeys(phase, rows) {
  const preferred = {
    "Strong scaling": ["gpus"],
    "Weak scaling": ["gpus", "global_batch_size"],
    "Distributed training strategy": ["strategy"],
    "Precision study": ["precision"],
    "Communication analysis": ["strategy"],
    "Baseline profiling": ["parameter_profile_target"],
    "Workload characterisation": ["batch_size", "parameter_sequence_length"],
  }[phase] || [];
  const memoryStudy = rows[0] && rows[0].parameter_memory_study_type;
  const memoryVariables = {
    micro_batch: ["parameter_micro_batch_size"],
    checkpointing: ["parameter_activation_checkpointing"],
    gradient_accumulation: ["parameter_gradient_accumulation_steps"],
  }[memoryStudy] || [];
  const focused = [...new Set([...memoryVariables, ...preferred])];
  if (focused.length) return focused.filter((key) => uniqueTrainingValues(rows, key).length > 1);
  const candidates = ["gpus", "strategy", "precision", "global_batch_size", "parameter_micro_batch_size", "parameter_gradient_accumulation_steps", "parameter_activation_checkpointing", "parameter_sequence_length"];
  return candidates.filter((key) => uniqueTrainingValues(rows, key).length > 1).slice(0, 3);
}

function trainingConstantFields(phase, rows) {
  const fields = ["platform_id", "comparability", "workload", "throughput_unit", "precision", "parameter_sequence_length"];
  if (phase === "Strong scaling") fields.push("global_batch_size");
  if (phase === "Weak scaling") fields.push("per_gpu_batch_size");
  if (phase === "Distributed training strategy") fields.push("gpus");
  if (phase === "Memory optimisation") fields.push("gpus", "strategy");
  return [...new Set(fields)].filter((key) => uniqueTrainingValues(rows, key).length === 1);
}

function uniqueTrainingValues(rows, key) {
  return [...new Set(rows.map((row) => String(trainingVariableValue(row, key) || "").trim()).filter(Boolean))];
}

function trainingVariableValue(row, key) {
  if (key === "parameter_micro_batch_size") return row.parameter_micro_batch_size || row.per_gpu_batch_size || row.batch_size;
  if (key === "parameter_sequence_length") return row.parameter_sequence_length || row.metric_sequence_length;
  return row[key];
}

function trainingVariableLabel(key) {
  return {
    gpus: "GPU count",
    strategy: "Distributed strategy",
    precision: "Numerical precision",
    global_batch_size: "Global batch size",
    per_gpu_batch_size: "Per-GPU batch size",
    batch_size: "Batch size",
    parameter_micro_batch_size: "Micro-batch size",
    parameter_gradient_accumulation_steps: "Accumulation steps",
    parameter_activation_checkpointing: "Activation checkpointing",
    parameter_sequence_length: "Sequence length",
    parameter_profile_target: "Profiler target",
  }[key] || displayName(key);
}

function trainingConfigurationLabel(row, keys) {
  if (!keys.length) return "Configuration " + row.experiment_id;
  return keys.map((key) => {
    const value = trainingVariableValue(row, key);
    if (key === "gpus") return value + " GPU" + (number(value) === 1 ? "" : "s");
    if (key === "strategy") return displayStrategy(value);
    if (key === "precision") return String(value || "").toUpperCase();
    if (key === "parameter_activation_checkpointing") return String(value).toLowerCase() === "true" ? "Checkpointing on" : "Checkpointing off";
    if (key === "global_batch_size") return "Global batch " + value;
    if (key === "batch_size") return "Batch " + value;
    if (key === "parameter_micro_batch_size") return "Micro-batch " + value;
    if (key === "parameter_gradient_accumulation_steps") return "Accumulate " + value + "×";
    if (key === "parameter_sequence_length") return "Sequence " + value;
    return String(value || row.experiment_id).replaceAll("_", " ");
  }).join(" · ");
}

function trainingPointMeta(row) {
  const trials = number(row.completed_trials);
  const mean = number(row.throughput_tokens_mean);
  const variation = mean > 0 && trials > 1 ? number(row.throughput_tokens_std) / mean * 100 : 0;
  return row.experiment_id + " · " + formatInteger(trials) + " completed trial" + (trials === 1 ? "" : "s") + (trials > 1 ? " · " + formatDecimal(variation, 1) + "% run variation" : " · interval unavailable");
}

function renderTrainingStudySummary(groups, study, message) {
  const container = document.getElementById("trainingStudySummary");
  container.replaceChildren();
  const points = groups.flatMap((group) => group.points.map((point) => ({ ...point, group })));
  if (!points.length) {
    container.append(emptyState(message || "No completed repeated training runs are available for this study."));
    return;
  }
  const fastest = [...points].sort((left, right) => right.value - left.value)[0];
  const repeatable = points.filter((point) => point.trials > 1).sort((left, right) => left.cvPercent - right.cvPercent)[0];
  const cards = [
    {
      label: "Question answered",
      value: trainingStudyQuestion(study),
      detail: "One controlled study is shown at a time.",
      wide: true,
    },
    {
      label: "Configurations compared",
      value: formatInteger(points.length),
      detail: groups.length + " separate platform/workload group" + (groups.length === 1 ? "" : "s"),
    },
    {
      label: "Fastest average",
      value: formatCompact(fastest.value),
      detail: fastest.group.title + " · " + fastest.label,
    },
    {
      label: "Most consistent runs",
      value: repeatable ? formatDecimal(repeatable.cvPercent, 1) + "% variation" : "Not estimable",
      detail: repeatable ? repeatable.group.title + " · " + repeatable.label : "At least two completed trials are required.",
    },
  ];
  cards.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "training-summary-card" + (entry.wide ? " training-summary-question" : "");
    card.innerHTML = "<span>" + escapeHtml(entry.label) + "</span><strong>" + escapeHtml(entry.value) + "</strong><small>" + escapeHtml(entry.detail) + "</small>";
    container.append(card);
  });
}

function trainingStudyQuestion(study) {
  return {
    "Strong scaling": "How much faster does a fixed workload run as GPU count increases?",
    "Weak scaling": "Does throughput keep pace as workload and GPU count increase together?",
    "Distributed training strategy": "Which distributed strategy delivers the highest throughput at the same GPU count?",
    "Precision study": "How does numerical precision change training throughput?",
    "Memory optimisation": "Which memory technique changes capacity without sacrificing too much throughput?",
    "Communication analysis": "How does the distributed strategy affect communication performance?",
    "Baseline profiling": "How consistent are the baseline profiler measurements?",
    "Workload characterisation": "How does the workload respond to its tested configuration settings?",
  }[study] || "How fast and repeatable are the completed training configurations?";
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
  card.classList.add("repeatability-chart-card");
  const width = 620;
  const rowHeight = 44;
  const height = Math.max(142, group.points.length * rowHeight + 54);
  const left = 205;
  const right = 42;
  const top = 18;
  const bottom = 34;
  const domain = paddedDomain(group.points.flatMap((point) => [point.low, point.high, point.value]), true);
  const scale = linearScale(domain[0], domain[1], left, width - right);
  const y = (index) => top + index * rowHeight + 14;
  const ticks = [domain[0], (domain[0] + domain[1]) / 2, domain[1]];
  const grids = ticks.map((value) => '<line class="svg-grid" x1="' + scale(value) + '" y1="' + (top - 8) + '" x2="' + scale(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label svg-axis-label" x="' + scale(value) + '" y="' + (height - 8) + '" text-anchor="middle">' + formatCompact(value) + "</text>").join("");
  const marks = group.points.map((point, index) => {
    const interval = point.hasInterval
      ? '<line class="svg-interval" x1="' + scale(point.low) + '" y1="' + y(index) + '" x2="' + scale(point.high) + '" y2="' + y(index) + '"/><line class="svg-interval-cap" x1="' + scale(point.low) + '" y1="' + (y(index) - 5) + '" x2="' + scale(point.low) + '" y2="' + (y(index) + 5) + '"/><line class="svg-interval-cap" x1="' + scale(point.high) + '" y1="' + (y(index) - 5) + '" x2="' + scale(point.high) + '" y2="' + (y(index) + 5) + '"/>'
      : "";
    const title = point.label + ": average " + formatInteger(point.value) + " " + humanizeThroughputUnit(group.unit) + (point.hasInterval ? "; 95% interval " + formatInteger(point.low) + "–" + formatInteger(point.high) : "; confidence interval unavailable");
    return '<text class="svg-label svg-condition-label" x="' + (left - 9) + '" y="' + (y(index) - 3) + '" text-anchor="end">' + escapeSvg(point.label) + '</text><text class="svg-label svg-condition-meta" x="' + (left - 9) + '" y="' + (y(index) + 11) + '" text-anchor="end">' + escapeSvg(point.meta) + "</text>" + interval + '<circle class="svg-point repeatability-point" ' + conditionTargetAttributes(point) + ' fill="' + colors[index % colors.length] + '" cx="' + scale(point.value) + '" cy="' + y(index) + '" r="5"><title>' + escapeSvg(title) + "</title></circle>";
  }).join("");
  const svg = svgElement(width, height, grids + marks);
  svg.setAttribute("aria-label", group.title + ": average throughput with 95% confidence intervals");
  card.append(svg);
  return card;
}

function renderStrategy() {
  const rows = filteredTrialRowsWithEvidence().filter((row) => row.mode === "training" && row.phase === "Distributed training strategy");
  const groups = buildStrategyComparisonGroups(rows, state.results);
  const selectedGroup = syncStrategyComparisonControl(groups);
  const metric = document.getElementById("strategyMetricFilter").value || "throughput";
  renderStrategyComparisonChart("strategyBars", selectedGroup, metric, "No matched distributed-strategy comparisons are available for the active filters.");
  const conditions = sum(groups.map((group) => group.entries.length));
  document.getElementById("strategyStatus").textContent = groups.length + " matched comparison" + (groups.length === 1 ? "" : "s") + " available · " + conditions + " configurations";
  markPanel("strategyBars", groups.length > 0, buildStrategyComparisonGroups(state.trialSummary, state.results).length > 0, "Distributed strategy", "trial_summary.csv", true);
}

function syncStrategyComparisonControl(groups) {
  return syncComparisonControl("strategyComparisonFilter", groups);
}

function syncComparisonControl(selectId, groups) {
  const select = document.getElementById(selectId);
  const current = select.value;
  select.replaceChildren();
  groups.forEach((group) => select.append(option(group.selectorLabel, group.id)));
  select.disabled = groups.length < 2;
  if (groups.some((group) => group.id === current)) select.value = current;
  else if (groups.length) select.value = groups[0].id;
  return groups.find((group) => group.id === select.value) || groups[0] || null;
}

function buildStrategyComparisonGroups(rows, evidenceRows = []) {
  return buildControlledTrainingGroups(
    rows,
    evidenceRows,
    (row) => row.mode === "training" && row.phase === "Distributed training strategy",
    (row) => [row.platform_id, row.comparability, row.workload, row.model, row.precision, row.gpus, row.global_batch_size, trainingVariableValue(row, "parameter_sequence_length"), row.throughput_unit],
    "strategy",
    "ddp",
  );
}

function renderPrecision() {
  const rows = filteredTrialRowsWithEvidence().filter((row) => row.mode === "training" && row.phase === "Precision study");
  const groups = buildPrecisionComparisonGroups(rows, state.results);
  const selectedGroup = syncComparisonControl("precisionComparisonFilter", groups);
  renderPrecisionComparison("precisionBars", selectedGroup, "No matched training-precision comparisons are available for the active filters.");
  const conditions = sum(groups.map((group) => group.entries.length));
  document.getElementById("precisionStatus").textContent = groups.length + " matched comparison" + (groups.length === 1 ? "" : "s") + " available · " + conditions + " configurations";
  markPanel("precisionBars", groups.length > 0, buildPrecisionComparisonGroups(state.trialSummary, state.results).length > 0, "Training precision", "trial_summary.csv", true);
}

function buildPrecisionComparisonGroups(rows, evidenceRows = []) {
  return buildControlledTrainingGroups(
    rows,
    evidenceRows,
    (row) => row.mode === "training" && row.phase === "Precision study",
    (row) => [row.platform_id, row.comparability, row.workload, row.model, row.mode, row.gpus, row.global_batch_size, trainingVariableValue(row, "parameter_sequence_length"), row.throughput_unit],
    "precision",
    "bf16",
  );
}

function renderPrecisionComparison(containerId, group, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!group) {
    container.append(emptyState(emptyMessage));
    return;
  }
  const context = document.createElement("div");
  context.className = "precision-context strategy-context";
  context.innerHTML = "<div><span>Comparison shown</span><strong>" + escapeHtml(group.title) + "</strong><small>" + escapeHtml(group.subtitle) + "</small></div>";
  context.append(comparisonContract(group.comparison));
  container.append(context);
  container.append(precisionSummary(group));
  container.append(precisionLegend(group.entries));
  const charts = document.createElement("div");
  charts.className = "precision-chart-grid";
  charts.append(precisionMetricChart(group, "throughput"));
  charts.append(precisionMetricChart(group, "memory"));
  container.append(charts);
  container.append(precisionTakeaway(group));
}

function precisionSummary(group) {
  const summary = document.createElement("div");
  summary.className = "precision-summary-grid";
  const fastest = [...group.entries].sort((left, right) => right.value - left.value)[0];
  const memoryEntries = group.entries.filter((entry) => entry.memory !== null);
  const lowestMemory = memoryEntries.length ? Math.min(...memoryEntries.map((entry) => entry.memory)) : null;
  const memoryLeaders = memoryEntries.filter((entry) => Math.abs(entry.memory - lowestMemory) < 0.001).map((entry) => entry.label);
  const bf16 = group.entries.find((entry) => entry.key === "bf16");
  const fp32 = group.entries.find((entry) => entry.key === "fp32");
  const bf16Advantage = bf16 && fp32 && fp32.value
    ? formatDecimal(bf16.value / fp32.value, 2) + "× FP32 throughput"
    : "Comparison unavailable";
  const bf16Memory = bf16 && fp32 && bf16.memory !== null && fp32.memory
    ? formatDecimal((1 - bf16.memory / fp32.memory) * 100, 1) + "% less memory"
    : "Memory comparison unavailable";
  const qualityEntries = group.entries.filter((entry) => entry.qualityLoss !== null);
  const qualityRange = qualityEntries.length
    ? formatDecimal(Math.min(...qualityEntries.map((entry) => entry.qualityLoss)), 3) + "–" + formatDecimal(Math.max(...qualityEntries.map((entry) => entry.qualityLoss)), 3)
    : "Not recorded";
  const cards = [
    ["Fastest format", fastest.label, formatInteger(fastest.value) + " " + humanizeThroughputUnit(group.unit)],
    ["Lowest GPU memory", memoryLeaders.length ? memoryLeaders.join(" and ") : "Not recorded", lowestMemory === null ? "No comparable memory evidence" : formatDecimal(lowestMemory, 2) + " GB measured"],
    ["BF16 compared with FP32", bf16Advantage, bf16Memory],
    ["Quality-loss sanity range", qualityRange, qualityEntries.length ? "Short-run loss across the displayed formats" : "No comparable quality-loss evidence"],
    ["Evidence in comparison", group.entries.length + " precision formats", sum(group.entries.map((entry) => entry.trials)) + " completed trials"],
  ];
  cards.forEach(([label, value, detail]) => {
    const card = document.createElement("article");
    card.innerHTML = "<span>" + escapeHtml(label) + "</span><strong>" + escapeHtml(value) + "</strong><small>" + escapeHtml(detail) + "</small>";
    summary.append(card);
  });
  return summary;
}

function precisionLegend(entries) {
  const legend = document.createElement("div");
  legend.className = "precision-legend strategy-legend";
  entries.forEach((entry) => {
    const item = document.createElement("span");
    item.innerHTML = '<i style="--strategy-color: ' + precisionColor(entry.key) + '"></i>' + escapeHtml(entry.label);
    legend.append(item);
  });
  return legend;
}

function precisionMetricChart(group, metric) {
  const card = document.createElement("article");
  card.className = "precision-metric-card";
  const isMemory = metric === "memory";
  card.innerHTML = "<div><span>" + (isMemory ? "Memory cost" : "Training speed") + "</span><h3>" + (isMemory ? "Measured GPU memory" : "Average training throughput") + "</h3><p>" + (isMemory ? "Shorter is better" : "Longer is better") + "</p></div>";
  const entries = group.entries.filter((entry) => !isMemory || entry.memory !== null);
  if (!entries.length) {
    card.append(emptyState("GPU memory was not recorded for this comparison."));
    return card;
  }
  const width = 620;
  const left = 82;
  const right = 118;
  const top = 20;
  const rowHeight = 54;
  const bottom = 40;
  const height = top + entries.length * rowHeight + bottom;
  const values = entries.map((entry) => isMemory ? entry.memory : entry.value);
  const maximum = max(values);
  const scale = linearScale(0, maximum || 1, left, width - right);
  const ticks = [0, maximum / 2, maximum];
  const formatValue = (value) => isMemory ? formatDecimal(value, 1) + " GB" : formatCompact(value);
  const grids = ticks.map((value) => '<line class="svg-grid" x1="' + scale(value) + '" y1="' + (top - 8) + '" x2="' + scale(value) + '" y2="' + (height - bottom + 4) + '"/><text class="svg-label svg-axis-label" x="' + scale(value) + '" y="' + (height - 12) + '" text-anchor="middle">' + escapeSvg(formatValue(value)) + "</text>").join("");
  const marks = entries.map((entry, index) => {
    const value = isMemory ? entry.memory : entry.value;
    const y = top + index * rowHeight;
    const detail = isMemory ? formatDecimal(value, 2) + " GB" : formatInteger(value);
    const fullDetail = detail + (isMemory ? " measured GPU memory" : " " + humanizeThroughputUnit(group.unit));
    return '<text class="svg-label precision-axis-name" x="' + (left - 10) + '" y="' + (y + 18) + '" text-anchor="end">' + escapeSvg(entry.label) + '</text><rect class="precision-svg-bar" ' + conditionTargetAttributes(entry) + ' fill="' + precisionColor(entry.key) + '" x="' + left + '" y="' + y + '" width="' + Math.max(3, scale(value) - left) + '" height="24" rx="4"><title>' + escapeSvg(entry.label + ": " + fullDetail) + '</title></rect><text class="svg-value precision-bar-value" x="' + (width - 8) + '" y="' + (y + 17) + '" text-anchor="end">' + escapeSvg(detail) + "</text>";
  }).join("");
  const svg = svgElement(width, height, grids + marks);
  svg.classList.add("precision-svg-chart");
  svg.setAttribute("aria-label", group.title + ": precision comparison by " + (isMemory ? "measured GPU memory" : "average training throughput"));
  card.append(svg);
  return card;
}

function precisionTakeaway(group) {
  const takeaway = document.createElement("p");
  takeaway.className = "precision-takeaway";
  const fastest = [...group.entries].sort((left, right) => right.value - left.value)[0];
  const bf16 = group.entries.find((entry) => entry.key === "bf16");
  if (!fastest || !bf16 || !bf16.value) {
    takeaway.textContent = "The selected comparison does not contain enough matched evidence for a relative interpretation.";
    return takeaway;
  }
  const speedRatio = fastest.value / bf16.value;
  const memoryPhrase = fastest.memory !== null && bf16.memory
    ? " and used " + formatDecimal(Math.abs((1 - fastest.memory / bf16.memory) * 100), 1) + "% " + (fastest.memory <= bf16.memory ? "less" : "more") + " measured GPU memory"
    : "";
  const qualityPhrase = fastest.qualityLoss !== null
    ? " Its short-run quality loss was " + formatDecimal(fastest.qualityLoss, 3) + "; interpret this as a sanity check, not a downstream quality verdict."
    : "";
  takeaway.innerHTML = "<strong>What this comparison shows:</strong> " + escapeHtml(fastest.label) + " was fastest at " + escapeHtml(formatDecimal(speedRatio, 2)) + "× BF16 throughput" + escapeHtml(memoryPhrase) + "." + escapeHtml(qualityPhrase);
  return takeaway;
}

function precisionColor(key) {
  return precisionColors[key] || "#64748b";
}

function buildControlledTrainingGroups(rows, evidenceRows, predicate, groupKey, variableKey, baselineValue) {
  const metadata = conditionMetadata(evidenceRows);
  const enriched = rows
    .filter((row) => predicate(row) && number(row.throughput_tokens_mean) > 0 && number(row.completed_trials) > 0)
    .map((row) => ({ ...(metadata.get(conditionId(row)) || {}), ...row }));
  return Object.entries(groupRows(enriched, groupKey)).map(([groupId, group]) => {
    const sample = group[0];
    const entries = group.map((row) => ({
      key: variableKey === "strategy" ? strategyVariantKey(row) : row[variableKey],
      baselineKey: row[variableKey],
      label: variableKey === "strategy" ? strategyVariantLabel(row) : String(row[variableKey] || "").toUpperCase(),
      value: number(row.throughput_tokens_mean),
      memory: conditionMetricAverage(evidenceRows, conditionId(row), ["metric_nvidia_smi_memory_used_gb_measured_region", "metric_memory_used_gb", "memory_used_gb"]),
      qualityLoss: conditionMetricAverage(evidenceRows, conditionId(row), ["metric_quality_loss", "metric_loss"]),
      trials: number(row.completed_trials),
      conditionId: conditionId(row),
    })).sort((left, right) => right.value - left.value);
    const baseline = entries.find((entry) => entry.baselineKey === baselineValue);
    entries.forEach((entry) => {
      entry.deltaPercent = baseline && baseline.value ? (entry.value / baseline.value - 1) * 100 : null;
      entry.isBaseline = entry === baseline;
    });
    return {
      id: group.map(conditionId).sort().join("::"),
      title: displayPlatform(sample.platform_id) + " · " + displayWorkload(sample.workload) + " · " + sample.gpus + " GPU" + (number(sample.gpus) === 1 ? "" : "s"),
      selectorLabel: displayPlatform(sample.platform_id) + " · " + displayWorkload(sample.workload) + " · " + sample.gpus + " GPU" + (number(sample.gpus) === 1 ? "" : "s"),
      subtitle: displayComparability(sample.comparability) + " · " + (variableKey === "strategy" ? String(sample.precision || "").toUpperCase() : "Training") + " · " + humanizeThroughputUnit(sample.throughput_unit),
      comparison: comparisonContext(group, [trainingVariableLabel(variableKey)], ["platform_id", "comparability", "workload", "precision", "gpus", "throughput_unit"]),
      unit: sample.throughput_unit,
      baselineLabel: baseline ? baseline.label : "",
      entries,
    };
  }).filter((group) => group.entries.length > 1).sort((left, right) => left.title.localeCompare(right.title));
}

function renderHardware() {
  const allGroups = buildBroadHardwareComparisonGroups(filteredTrialRowsWithEvidence(), state.results);
  setHardwareScopeOptions("hardwarePhaseFilter", uniqueStrings(allGroups.map((group) => group.phase)), "All studies");
  setHardwareScopeOptions("hardwareWorkloadFilter", uniqueStrings(allGroups.map((group) => group.workload)), "All workloads", displayWorkload);
  setHardwareScopeOptions("hardwareModeFilter", uniqueStrings(allGroups.map((group) => group.mode)), "Training and inference", displayHardwareMode);
  const phase = document.getElementById("hardwarePhaseFilter").value;
  const workload = document.getElementById("hardwareWorkloadFilter").value;
  const mode = document.getElementById("hardwareModeFilter").value;
  const groups = allGroups.filter((group) =>
    (!phase || group.phase === phase) &&
    (!workload || group.workload === workload) &&
    (!mode || group.mode === mode)
  );
  const landscape = buildHardwareLandscape(groups);
  renderHardwareLandscape("hardwareLandscape", landscape, "No exact H100/L40S protocol pair matches the active filters.");
  const selectedGroup = syncComparisonControl("hardwareComparisonFilter", groups);
  renderHardwareComparison("hardwareComparison", selectedGroup, "No matched H100 and L40S hardware comparison is available for the active filters.");
  const scope = groups.length === allGroups.length ? "" : " shown of " + allGroups.length;
  document.getElementById("hardwareStatus").textContent = groups.length
    ? groups.length + scope + " matched protocols · " + sum(groups.map((group) => sum(group.entries.map((entry) => entry.trials)))) + " trial observations"
    : "No matched hardware evidence";
  markPanel("hardwareComparison", groups.length > 0, buildBroadHardwareComparisonGroups(state.trialSummary, state.results).length > 0, "Hardware comparison", "trial_summary.csv", true);
}

function setHardwareScopeOptions(selectId, values, allLabel, display = (value) => value) {
  const select = document.getElementById(selectId);
  const current = select.value;
  select.replaceChildren(option(allLabel, ""));
  values.forEach((value) => select.append(option(display(value), value)));
  select.value = values.includes(current) ? current : "";
}

function buildHardwareComparisonGroups(rows) {
  const eligible = rows.filter((row) => row.comparability === "exact" && number(row.avg_throughput_tokens_sec) > 0);
  const grouped = groupRows(eligible, (row) => [row.comparability, row.workload, row.model, row.mode, row.precision, row.throughput_unit]);
  return Object.entries(grouped).map(([groupId, group]) => {
    const platforms = new Set(group.map((row) => row.platform_id));
    if (!platforms.has("h100") || !platforms.has("l40s")) return null;
    const sample = group[0];
    const entries = group.map((row) => ({
      key: row.platform_id,
      label: displayPlatform(row.platform_id),
      value: number(row.avg_throughput_tokens_sec),
      memory: isNumeric(row.avg_memory_used_gb) ? number(row.avg_memory_used_gb) : null,
      latency: isNumeric(row.avg_latency_p50_ms) ? number(row.avg_latency_p50_ms) : null,
      trials: number(row.trials),
      conditionId: conditionId(row),
    })).sort((left, right) => right.value - left.value);
    return {
      id: groupId.split("\u0000").map((part) => encodeURIComponent(part)).join("::"),
      title: displayWorkload(sample.workload) + " · " + (sample.mode === "training" ? "Training" : "Inference"),
      selectorLabel: displayWorkload(sample.workload) + " · " + (sample.mode === "training" ? "Training" : "Inference"),
      subtitle: String(sample.precision || "").toUpperCase() + " · " + displayComparability(sample.comparability) + " · " + humanizeThroughputUnit(sample.throughput_unit),
      comparison: comparisonContext(group, ["GPU platform"], ["comparability", "workload", "model", "mode", "precision", "throughput_unit"]),
      workload: sample.workload,
      mode: sample.mode,
      unit: sample.throughput_unit,
      entries,
    };
  }).filter(Boolean).sort((left, right) => left.title.localeCompare(right.title));
}

function buildBroadHardwareComparisonGroups(trialRows, evidenceRows = []) {
  const metadata = conditionMetadata(evidenceRows);
  const eligible = trialRows
    .filter((row) => row.comparability === "exact" && number(row.completed_trials) > 0 && number(row.throughput_tokens_mean) > 0)
    .map((row) => ({ ...(metadata.get(conditionId(row)) || {}), ...row }));
  const grouped = groupRows(eligible, hardwareProtocolSignature);
  return Object.values(grouped).map((group) => {
    const platformRows = Object.fromEntries(group.map((row) => [row.platform_id, row]));
    if (group.length !== 2 || !platformRows.h100 || !platformRows.l40s) return null;
    const sample = group[0];
    const experimentIds = uniqueStrings(group.map((row) => row.experiment_id));
    const entries = group.map((row) => ({
      key: row.platform_id,
      label: displayPlatform(row.platform_id),
      value: number(row.throughput_tokens_mean),
      runtime: isNumeric(row.runtime_mean_seconds) ? number(row.runtime_mean_seconds) : null,
      memory: conditionMetricAverage(evidenceRows, conditionId(row), ["metric_nvidia_smi_memory_used_gb_measured_region", "metric_memory_used_gb", "memory_used_gb"]),
      latency: conditionMetricAverage(evidenceRows, conditionId(row), ["metric_latency_p50_ms", "latency_p50_ms"]),
      latencyP95: conditionMetricAverage(evidenceRows, conditionId(row), ["metric_latency_p95_ms", "latency_p95_ms"]),
      energy: conditionEnergyAverage(evidenceRows, conditionId(row)),
      trials: number(row.completed_trials),
      conditionId: conditionId(row),
    })).sort((left, right) => right.value - left.value);
    return {
      id: group.map(conditionId).sort().join("::"),
      title: displayWorkload(sample.workload) + " · " + displayHardwareMode(sample.mode),
      selectorLabel: sample.phase + " · " + displayWorkload(sample.workload) + " · " + hardwareConfigurationLabel(sample) + " · " + experimentIds.join("/") ,
      subtitle: sample.phase + " · " + hardwareConfigurationLabel(sample) + " · " + humanizeThroughputUnit(sample.throughput_unit),
      comparison: comparisonContext(group, ["GPU platform"], ["comparability", "workload", "model", "mode", "precision", "gpus", "strategy", "framework", "throughput_unit"]),
      comparability: sample.comparability,
      experimentIds,
      phase: sample.phase,
      workload: sample.workload,
      mode: sample.mode,
      precision: sample.precision,
      strategy: sample.strategy,
      gpus: number(sample.gpus),
      unit: sample.throughput_unit,
      entries,
    };
  }).filter(Boolean).sort((left, right) => (left.phase + left.selectorLabel).localeCompare(right.phase + right.selectorLabel));
}

function hardwareProtocolSignature(row) {
  return hardwareProtocolFields.map((key) => String(row[key] || "").trim());
}

function hardwareConfigurationLabel(row) {
  const parts = [
    number(row.gpus) + " GPU" + (number(row.gpus) === 1 ? "" : "s"),
    String(row.precision || "unspecified precision").toUpperCase(),
  ];
  if (row.strategy && row.strategy !== "none") parts.push(displayStrategy(row.strategy));
  const framework = row.parameter_framework_label || row.parameter_framework || row.framework;
  if (framework && framework !== "none") parts.push(String(framework).replaceAll("_", " "));
  return parts.join(" · ");
}

function displayHardwareMode(mode) {
  return mode === "training" ? "Training" : mode === "inference" ? "Inference" : String(mode || "Unspecified mode");
}

function conditionEnergyAverage(rows, id) {
  const values = rows.filter((row) =>
    row.status === "completed" &&
    conditionId(row) === id &&
    row.energy_scope === "measured_region" &&
    row.energy_measurement_status === "sufficient" &&
    number(row.energy_joules) > 0
  ).map((row) => number(row.energy_joules));
  return values.length ? average(values) : null;
}

function buildHardwareLandscape(groups) {
  const comparisons = groups.map((group) => {
    const ratio = hardwareThroughputRatio(group);
    return ratio ? { ...ratio, group } : null;
  }).filter(Boolean);
  if (!comparisons.length) return null;
  const classify = (ratio) => ratio > 1.02 ? "h100" : ratio < 0.98 ? "l40s" : "parity";
  const summarize = (items, label) => {
    const ratios = items.map((item) => item.ratio);
    return {
      label,
      comparisons: items.length,
      trials: sum(items.flatMap((item) => item.group.entries.map((entry) => entry.trials))),
      medianRatio: median(ratios),
      minimumRatio: Math.min(...ratios),
      maximumRatio: Math.max(...ratios),
      h100Leads: items.filter((item) => classify(item.ratio) === "h100").length,
      l40sLeads: items.filter((item) => classify(item.ratio) === "l40s").length,
      parity: items.filter((item) => classify(item.ratio) === "parity").length,
      conditionIds: uniqueStrings(items.flatMap((item) => item.group.entries.map((entry) => entry.conditionId))),
    };
  };
  const phases = Object.entries(groupRows(comparisons, (item) => [item.group.phase || "Unclassified study"]))
    .map(([phase, items]) => summarize(items, phase))
    .sort((left, right) => right.comparisons - left.comparisons || left.label.localeCompare(right.label));
  const modes = Object.entries(groupRows(comparisons, (item) => [displayHardwareMode(item.group.mode)]))
    .map(([mode, items]) => summarize(items, mode));
  return {
    ...summarize(comparisons, "All matched protocols"),
    items: comparisons,
    phases,
    modes,
    strongestH100: comparisons.filter((item) => item.ratio > 1.02).sort((left, right) => right.ratio - left.ratio).slice(0, 3),
    strongestL40S: comparisons.filter((item) => item.ratio < 0.98).sort((left, right) => left.ratio - right.ratio).slice(0, 3),
  };
}

function renderHardwareLandscape(containerId, landscape, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!landscape) {
    container.append(emptyState(emptyMessage));
    return;
  }
  const summary = document.createElement("div");
  summary.className = "hardware-campaign-summary";
  const cards = [
    ["Matched protocols", formatInteger(landscape.comparisons), formatInteger(landscape.trials) + " completed trial observations"],
    ["Typical H100/L40S ratio", formatDecimal(landscape.medianRatio, 2) + "×", "Median of controlled, unitless throughput ratios"],
    ["H100 leads", formatInteger(landscape.h100Leads), formatDecimal(landscape.h100Leads / landscape.comparisons * 100, 0) + "% of matched protocols"],
    ["L40S leads", formatInteger(landscape.l40sLeads), formatInteger(landscape.parity) + " additional protocol" + (landscape.parity === 1 ? "" : "s") + " near parity"],
  ];
  cards.forEach(([label, value, detail]) => {
    const card = document.createElement("article");
    card.innerHTML = "<span>" + escapeHtml(label) + "</span><strong>" + escapeHtml(value) + "</strong><small>" + escapeHtml(detail) + "</small>";
    summary.append(card);
  });
  container.append(summary);

  const modeSummary = document.createElement("div");
  modeSummary.className = "hardware-mode-summary";
  landscape.modes.forEach((mode) => {
    const item = document.createElement("article");
    item.innerHTML = "<div><span>" + escapeHtml(mode.label) + "</span><strong>" + escapeHtml(formatDecimal(mode.medianRatio, 2) + "× median") + "</strong></div><p>H100 led " + formatInteger(mode.h100Leads) + ", L40S led " + formatInteger(mode.l40sLeads) + ", and " + formatInteger(mode.parity) + " were near parity across " + formatInteger(mode.comparisons) + " matched protocols.</p>";
    modeSummary.append(item);
  });
  container.append(modeSummary);

  const body = document.createElement("div");
  body.className = "hardware-landscape-grid";
  body.append(hardwarePhaseLandscape(landscape.phases));
  body.append(hardwareContrastList(landscape));
  container.append(body);
}

function hardwarePhaseLandscape(phases) {
  const card = document.createElement("article");
  card.className = "hardware-landscape-card";
  card.innerHTML = '<div class="hardware-landscape-heading"><div><span>Study-level synthesis</span><h3>Where does each platform lead?</h3></div><div class="ratio-direction"><span>L40S faster</span><i></i><span>H100 faster</span></div></div>';
  const ratios = phases.flatMap((phase) => [phase.minimumRatio, phase.maximumRatio]);
  const extent = Math.max(1, ...ratios.map((ratio) => Math.abs(Math.log2(ratio))));
  const rows = document.createElement("div");
  rows.className = "hardware-ratio-rows";
  phases.forEach((phase) => {
    const row = document.createElement("div");
    row.className = "hardware-ratio-row";
    const low = hardwareRatioPosition(phase.minimumRatio, extent);
    const high = hardwareRatioPosition(phase.maximumRatio, extent);
    const middle = hardwareRatioPosition(phase.medianRatio, extent);
    row.innerHTML = '<div class="hardware-ratio-label"><strong>' + escapeHtml(phase.label) + '</strong><span>' + formatInteger(phase.comparisons) + " protocol" + (phase.comparisons === 1 ? "" : "s") + " · " + formatInteger(phase.trials) + ' trials</span></div><div class="hardware-ratio-track"><i class="ratio-parity"></i><i class="ratio-range" style="left:' + low + "%;width:" + Math.max(1, high - low) + '%"></i><i class="ratio-median" style="left:' + middle + '%"></i></div><div class="hardware-ratio-value"><strong>' + formatDecimal(phase.medianRatio, 2) + '×</strong><span>' + formatDecimal(phase.minimumRatio, 2) + "–" + formatDecimal(phase.maximumRatio, 2) + "× range</span></div>";
    rows.append(row);
  });
  card.append(rows);
  const note = document.createElement("p");
  note.className = "hardware-landscape-note";
  note.textContent = "Dots are median H100/L40S throughput ratios; lines show the full range of matched conditions in each study. The logarithmic axis gives equal visual weight to reciprocal advantages around 1× parity.";
  card.append(note);
  return card;
}

function hardwareRatioPosition(ratio, extent) {
  return Math.max(0, Math.min(100, 50 + Math.log2(ratio) / (2 * extent) * 100));
}

function hardwareContrastList(landscape) {
  const card = document.createElement("article");
  card.className = "hardware-landscape-card hardware-contrast-card";
  card.innerHTML = "<div class=\"hardware-landscape-heading\"><div><span>Largest observed differences</span><h3>Protocols worth inspecting</h3></div></div>";
  const sections = [
    ["Strongest H100 advantages", landscape.strongestH100],
    ["Strongest L40S advantages", landscape.strongestL40S],
  ];
  sections.forEach(([title, items]) => {
    const section = document.createElement("section");
    const heading = document.createElement("h4");
    heading.textContent = title;
    section.append(heading);
    if (!items.length) {
      const note = document.createElement("p");
      note.className = "hardware-contrast-empty";
      note.textContent = "No matched protocol in this filtered view crossed the 2% lead threshold.";
      section.append(note);
    }
    items.forEach((item) => {
      const row = document.createElement("article");
      row.className = "hardware-contrast-row";
      setConditionTarget(row, "", item.group.entries.map((entry) => entry.conditionId));
      row.innerHTML = "<div><strong>" + escapeHtml(item.group.experimentIds.join(" / ") + " · " + item.group.phase) + "</strong><span>" + escapeHtml(displayWorkload(item.group.workload) + " · " + displayHardwareMode(item.group.mode) + " · " + hardwareGroupConfigurationLabel(item.group)) + "</span></div><b>" + escapeHtml(formatDecimal(item.ratio, 2) + "×") + "</b>";
      section.append(row);
    });
    card.append(section);
  });
  return card;
}

function hardwareGroupConfigurationLabel(group) {
  const parts = [group.gpus + " GPU" + (group.gpus === 1 ? "" : "s"), String(group.precision || "").toUpperCase()];
  if (group.strategy && group.strategy !== "none") parts.push(displayStrategy(group.strategy));
  return parts.filter(Boolean).join(" · ");
}

function renderHardwareComparison(containerId, group, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!group) {
    container.append(emptyState(emptyMessage));
    return;
  }
  const context = document.createElement("div");
  context.className = "hardware-context strategy-context";
  context.innerHTML = "<div><span>Controlled comparison</span><strong>" + escapeHtml(group.title) + "</strong><small>" + escapeHtml(group.subtitle) + "</small></div>";
  context.append(comparisonContract(group.comparison));
  container.append(context);
  container.append(hardwareComparisonSummary(group));
  const legend = document.createElement("div");
  legend.className = "hardware-legend strategy-legend";
  group.entries.forEach((entry) => {
    const item = document.createElement("span");
    item.innerHTML = '<i style="--strategy-color: ' + hardwareColor(entry.key) + '"></i>' + escapeHtml(entry.label);
    legend.append(item);
  });
  container.append(legend);
  const charts = document.createElement("div");
  charts.className = "hardware-chart-grid";
  const metrics = ["throughput", "runtime", "memory"];
  if (group.mode === "inference") metrics.push("latency");
  if (group.entries.every((entry) => entry.energy !== null)) metrics.push("energy");
  metrics.filter((metric) => metric === "throughput" || group.entries.every((entry) => entry[metric] !== null)).forEach((metric) => {
    charts.append(hardwareMetricChart(group, metric));
  });
  container.append(charts);
}

function hardwareComparisonSummary(group) {
  const summary = document.createElement("div");
  summary.className = "hardware-summary-grid";
  const fastest = group.entries[0];
  const h100 = group.entries.find((entry) => entry.key === "h100");
  const l40s = group.entries.find((entry) => entry.key === "l40s");
  const ratio = h100 && l40s && l40s.value ? h100.value / l40s.value : null;
  const secondaryKey = group.mode === "inference" ? "latency" : "memory";
  const secondary = group.entries.filter((entry) => entry[secondaryKey] !== null).sort((left, right) => left[secondaryKey] - right[secondaryKey])[0];
  const cards = [
    ["Throughput leader", fastest.label, formatInteger(fastest.value) + " " + humanizeThroughputUnit(group.unit)],
    ["H100 / L40S throughput", ratio === null ? "Not estimable" : formatDecimal(ratio, 2) + "×", ratio === null ? "Both platforms are required" : ratio >= 1 ? "H100 relative to the matched L40S run" : "L40S was faster in this matched condition"],
    [group.mode === "inference" ? "Lowest p50 latency" : "Lowest GPU memory", secondary ? secondary.label : "Not recorded", secondary ? formatDecimal(secondary[secondaryKey], 2) + (secondaryKey === "latency" ? " ms" : " GB") : "No secondary metric"],
    ["Evidence in comparison", sum(group.entries.map((entry) => entry.trials)) + " trials", group.entries.length + " matched platform conditions"],
  ];
  cards.forEach(([label, value, detail]) => {
    const card = document.createElement("article");
    card.innerHTML = "<span>" + escapeHtml(label) + "</span><strong>" + escapeHtml(value) + "</strong><small>" + escapeHtml(detail) + "</small>";
    summary.append(card);
  });
  return summary;
}

function hardwareMetricChart(group, metric) {
  const card = document.createElement("article");
  card.className = "hardware-metric-card";
  const labels = {
    throughput: ["Performance", "Average throughput", "Higher is better"],
    runtime: ["Completion time", "Average runtime", "Lower is better"],
    memory: ["Memory footprint", "Average measured GPU memory", "Lower is better"],
    latency: ["Responsiveness", "Median request latency", "Lower is better"],
    energy: ["Energy to complete", "Measured-region energy", "Lower is better for this fixed protocol"],
  }[metric];
  card.innerHTML = "<div><span>" + labels[0] + "</span><h3>" + labels[1] + "</h3><p>" + labels[2] + "</p></div>";
  const key = metric === "throughput" ? "value" : metric;
  const entries = group.entries.filter((entry) => entry[key] !== null);
  const maximum = max(entries.map((entry) => entry[key])) || 1;
  const bars = document.createElement("div");
  bars.className = "hardware-bars";
  entries.forEach((entry) => {
    const value = entry[key];
    const row = document.createElement("article");
    row.className = "hardware-bar-row";
    setConditionTarget(row, entry.conditionId);
    const formatted = metric === "throughput"
      ? formatInteger(value)
      : formatDecimal(value, 2) + ({ memory: " GB", latency: " ms", runtime: " s", energy: " J" }[metric] || "");
    row.innerHTML = '<div><strong>' + escapeHtml(entry.label) + "</strong><span>" + escapeHtml(formatted) + '</span></div><div class="hardware-bar-track"><i style="width:' + Math.max(3, value / maximum * 100) + "%;--hardware-color:" + hardwareColor(entry.key) + '"></i></div>';
    bars.append(row);
  });
  card.append(bars);
  return card;
}

function hardwareColor(key) {
  return hardwareColors[key] || "#64748b";
}

function renderStrategyComparisonChart(containerId, group, metric, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!group) {
    container.append(emptyState(emptyMessage));
    return;
  }
  const entries = group.entries.filter((entry) => metric === "throughput" || entry.memory !== null);
  if (!entries.length) {
    container.append(emptyState("GPU memory was not recorded for this matched comparison."));
    return;
  }
  const context = document.createElement("div");
  context.className = "strategy-context";
  context.innerHTML = "<div><span>Comparison shown</span><strong>" + escapeHtml(group.title) + "</strong><small>" + escapeHtml(group.subtitle) + "</small></div>";
  context.append(comparisonContract(group.comparison));
  container.append(context);
  container.append(strategySummary(group));
  container.append(strategyLegend(group.entries));
  container.append(strategyBarChart(group, entries, metric));
  const note = document.createElement("p");
  note.className = "strategy-chart-note";
  note.textContent = metric === "throughput"
    ? "Longer bars indicate higher average training throughput. Percentages are relative to the matched DDP baseline."
    : "Shorter bars indicate lower average measured GPU memory. Throughput results remain available through the metric selector.";
  container.append(note);
}

function strategySummary(group) {
  const summary = document.createElement("div");
  summary.className = "strategy-summary-grid";
  const fastest = [...group.entries].sort((left, right) => right.value - left.value)[0];
  const lowestMemory = group.entries.filter((entry) => entry.memory !== null).sort((left, right) => left.memory - right.memory)[0];
  const totalTrials = sum(group.entries.map((entry) => entry.trials));
  const cards = [
    ["Highest throughput", fastest.label, formatInteger(fastest.value) + " " + humanizeThroughputUnit(group.unit) + (fastest.isBaseline ? " · DDP baseline" : fastest.deltaPercent === null ? "" : " · " + signedPercent(fastest.deltaPercent) + " vs " + group.baselineLabel)],
    ["Lowest GPU memory", lowestMemory ? lowestMemory.label : "Not recorded", lowestMemory ? formatDecimal(lowestMemory.memory, 2) + " GB measured" : "No comparable memory evidence"],
    ["Evidence in chart", group.entries.length + " strategies", totalTrials + " completed trials"],
  ];
  cards.forEach(([label, value, detail]) => {
    const card = document.createElement("article");
    card.innerHTML = "<span>" + escapeHtml(label) + "</span><strong>" + escapeHtml(value) + "</strong><small>" + escapeHtml(detail) + "</small>";
    summary.append(card);
  });
  return summary;
}

function strategyLegend(entries) {
  const legend = document.createElement("div");
  legend.className = "strategy-legend";
  entries.forEach((entry) => {
    const item = document.createElement("span");
    item.innerHTML = '<i style="--strategy-color: ' + strategyColor(entry.key) + '"></i>' + escapeHtml(entry.label);
    legend.append(item);
  });
  return legend;
}

function strategyBarChart(group, entries, metric) {
  const width = 760;
  const rowHeight = 52;
  const left = 165;
  const right = 110;
  const top = 24;
  const bottom = 42;
  const height = Math.max(150, top + entries.length * rowHeight + bottom);
  const values = entries.map((entry) => metric === "memory" ? entry.memory : entry.value);
  const maximum = max(values);
  const scale = linearScale(0, maximum || 1, left, width - right);
  const ticks = [0, maximum / 2, maximum];
  const formatValue = (value) => metric === "memory" ? formatDecimal(value, 1) + " GB" : formatCompact(value);
  const grids = ticks.map((value) => '<line class="svg-grid" x1="' + scale(value) + '" y1="' + (top - 8) + '" x2="' + scale(value) + '" y2="' + (height - bottom + 4) + '"/><text class="svg-label svg-axis-label" x="' + scale(value) + '" y="' + (height - 12) + '" text-anchor="middle">' + escapeSvg(formatValue(value)) + "</text>").join("");
  const marks = entries.map((entry, index) => {
    const value = metric === "memory" ? entry.memory : entry.value;
    const y = top + index * rowHeight;
    const barWidth = Math.max(3, scale(value) - left);
    const detail = metric === "memory"
      ? formatDecimal(value, 2) + " GB · " + formatInteger(entry.value) + " " + humanizeThroughputUnit(group.unit)
      : formatInteger(value) + " · " + (entry.isBaseline ? "DDP baseline" : entry.deltaPercent === null ? "No DDP baseline" : signedPercent(entry.deltaPercent) + " vs " + group.baselineLabel);
    const visibleDetail = metric === "memory"
      ? formatDecimal(value, 2) + " GB"
      : formatCompact(value) + (entry.isBaseline ? " · baseline" : entry.deltaPercent === null ? "" : " · " + signedPercent(entry.deltaPercent));
    return '<text class="svg-label strategy-axis-name" x="' + (left - 10) + '" y="' + (y + 17) + '" text-anchor="end">' + escapeSvg(entry.label) + '</text><rect class="strategy-svg-bar" ' + conditionTargetAttributes(entry) + ' fill="' + strategyColor(entry.key) + '" x="' + left + '" y="' + y + '" width="' + barWidth + '" height="22" rx="4"><title>' + escapeSvg(entry.label + ": " + detail) + '</title></rect><text class="svg-value strategy-bar-value" x="' + (width - 8) + '" y="' + (y + 15) + '" text-anchor="end">' + escapeSvg(visibleDetail) + "</text>";
  }).join("");
  const svg = svgElement(width, height, grids + marks);
  svg.classList.add("strategy-svg-chart");
  svg.setAttribute("aria-label", group.title + ": strategy comparison by " + (metric === "memory" ? "GPU memory" : "training throughput"));
  return svg;
}

function strategyColor(key) {
  return strategyColors[key] || "#64748b";
}

function strategyVariantKey(row) {
  if (row.strategy !== "fsdp") return row.strategy;
  const detail = row.parameter_strategy_detail || row.metric_strategy_detail_observed || "";
  return detail ? "fsdp_" + detail : "fsdp";
}

function strategyVariantLabel(row) {
  if (row.strategy !== "fsdp") return displayStrategy(row.strategy);
  const detail = row.parameter_strategy_detail || row.metric_strategy_detail_observed || "";
  return {
    full_shard: "FSDP · full shard",
    sharded_grad_op: "FSDP · sharded gradients",
  }[detail] || "FSDP";
}

function conditionMetricAverage(rows, id, keys) {
  const values = rows.filter((row) => row.status === "completed" && conditionId(row) === id).flatMap((row) => {
    const value = firstRecorded(row, keys);
    return hasValue(value) && Number.isFinite(Number.parseFloat(value)) ? [number(value)] : [];
  });
  return values.length ? average(values) : null;
}

function renderControlledComparisonCards(containerId, groups, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState(emptyMessage));
    return;
  }
  groups.forEach((group) => {
    const card = document.createElement("article");
    card.className = "controlled-comparison-card";
    card.innerHTML = "<h3>" + escapeHtml(group.title) + "</h3><p>" + escapeHtml(group.subtitle) + "</p>";
    card.append(comparisonContract(group.comparison));
    const bars = document.createElement("div");
    bars.className = "controlled-bars";
    const maximum = max(group.entries.map((entry) => entry.value));
    group.entries.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "controlled-bar-row";
      setConditionTarget(row, entry.conditionId);
      const relative = entry.isBaseline ? "Baseline" : entry.deltaPercent === null ? "No baseline" : signedPercent(entry.deltaPercent) + " vs " + group.baselineLabel;
      const memory = entry.memory === null ? "memory not recorded" : formatDecimal(entry.memory, 2) + " GB memory";
      row.innerHTML = '<div class="bar-meta"><strong>' + escapeHtml(entry.label) + '</strong><span>' + escapeHtml(formatInteger(entry.value) + " " + humanizeThroughputUnit(group.unit)) + '</span></div><div class="bar-detail">' + escapeHtml(relative + " · " + memory + " · " + entry.trials + " trials") + '</div><div class="bar-track"><div class="bar-fill" style="width: ' + Math.max(3, entry.value / maximum * 100) + '%"></div></div>';
      bars.append(row);
    });
    card.append(bars);
    container.append(card);
  });
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
  const scalingType = document.getElementById("scalingTypeFilter").value;
  const series = filterScalingSeries(buildScalingSeries(result.rows), scalingType);
  const allSeriesKeys = buildScalingSeries(state.scalingSummary).map((item) => item.key);
  series.forEach((item) => item.points.forEach((point) => {
    point.conditionIds = matchingConditionIds({ platform_id: item.platform, comparability: item.comparability, workload: item.workload, strategy: item.strategy, precision: item.precision, scaling_type: item.scalingType, gpus: point.x, throughput_unit: item.unit });
  }));
  series.forEach((item) => {
    item.conditionIds = [...new Set(item.points.flatMap((point) => point.conditionIds || []))];
    item.color = scalingColors[Math.max(0, allSeriesKeys.indexOf(item.key)) % scalingColors.length];
  });
  const emptyMessage = scalingType
    ? "No completed " + scalingType + "-scaling evidence matches the active filters."
    : "No completed scaling-study evidence matches the active filters.";
  renderCombinedScalingChart("scalingCharts", series, { empty: result.message || emptyMessage });
  const pointCount = sum(series.map((item) => item.points.length));
  const scope = scalingType ? scalingType + " scaling" : "strong and weak scaling";
  document.getElementById("scalingStatus").textContent = result.message || series.length + " colour-coded series · " + pointCount + " points · " + scope;
  markPanel("scalingCharts", series.length > 0, buildScalingSeries(state.scalingSummary).length > 0, "Scaling speedup", "scaling_summary.csv", true);
}

function filterScalingSeries(series, scalingType) {
  return scalingType ? series.filter((item) => item.scalingType === scalingType) : [...series];
}

function buildScalingSeries(rows) {
  const grouped = groupRows(rows.filter((row) => number(row.gpus) > 0 && number(row.speedup) > 0), (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.strategy || "unspecified", row.precision || "unspecified", row.throughput_unit || "unit_not_recorded", row.scaling_type || "unspecified"]);
  return Object.entries(grouped).map(([key, group]) => {
    const [platform, comparability, workload, strategy, precision, unit, scalingType] = key.split("\u0000");
    return {
      key,
      title: platform + " · " + comparability + " · " + workload,
      subtitle: strategy + "; " + precision + "; " + scalingType + "; " + unit,
      platform, comparability, workload, strategy, precision, unit, scalingType,
      comparison: comparisonContext(group, ["GPU count"], ["platform_id", "comparability", "workload", "strategy", "precision", "throughput_unit", "scaling_type"]),
      points: group.map((row) => ({ x: number(row.gpus), y: number(row.speedup), label: row.gpus + " GPU: " + formatDecimal(number(row.speedup), 2) + "× speedup; " + formatDecimal(number(row.scaling_efficiency) * 100, 1) + "% efficiency", conditionId: conditionId(row) })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderCombinedScalingChart(containerId, series, options) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!series.length) {
    container.append(emptyState(options.empty));
    return;
  }
  container.append(combinedScalingChartCard(series));
}

function combinedScalingChartCard(series) {
  const card = document.createElement("article");
  card.className = "combined-scaling-card";
  const heading = document.createElement("div");
  heading.className = "combined-scaling-heading";
  heading.innerHTML = "<div><h3>Scaling trajectories across recorded studies</h3><p>Speedup is relative to the smallest GPU-count baseline within each coloured series.</p></div>";
  heading.append(comparisonContract({
    changing: ["GPU count within each line"],
    constants: [{ label: "within a line", value: "platform, workload, precision, strategy, scaling type, and throughput unit" }],
  }));
  card.append(heading);

  const legend = document.createElement("div");
  legend.className = "scaling-series-legend";
  series.forEach((item) => legend.append(scalingLegendItem(item)));
  card.append(legend);
  card.append(combinedScalingSvg(series));
  return card;
}

function scalingLegendItem(series) {
  const item = document.createElement("button");
  item.type = "button";
  item.className = "scaling-legend-item";
  item.style.setProperty("--series-color", series.color);
  if (series.scalingType === "weak") item.classList.add("is-weak");
  setConditionTarget(item, "", series.conditionIds);
  const experimentIds = [...new Set(series.conditionIds.map((id) => id.split("@")[0]))];
  const baseline = series.points[0]?.x;
  item.innerHTML = '<i aria-hidden="true"></i><span><strong>' + escapeHtml([
    series.platform.toUpperCase(), displayWorkload(series.workload), series.scalingType + " scaling",
  ].join(" · ")) + '</strong><small>' + escapeHtml([
    experimentIds.join(" / ") || "Recorded series", series.comparability, displayStrategy(series.strategy), series.precision.toUpperCase(),
    baseline ? "baseline " + baseline + " GPU" + (baseline === 1 ? "" : "s") : "",
  ].filter(Boolean).join(" · ")) + "</small></span>";
  return item;
}

function combinedScalingSvg(series) {
  const width = 920;
  const height = 400;
  const left = 64;
  const right = 28;
  const top = 24;
  const bottom = 48;
  const xValues = [...new Set(series.flatMap((item) => item.points.map((point) => point.x)))].sort((a, b) => a - b);
  const xPosition = new Map(xValues.map((value, index) => [
    value,
    xValues.length === 1 ? (left + width - right) / 2 : left + index * ((width - right - left) / (xValues.length - 1)),
  ]));
  const observedMaximum = max(series.flatMap((item) => item.points.map((point) => point.y)));
  const yMaximum = Math.max(2, Math.ceil(observedMaximum));
  const y = linearScale(0, yMaximum, height - bottom, top);
  const yTicks = Array.from({ length: 5 }, (_, index) => index * yMaximum / 4);
  const verticalGrid = xValues.map((value) => '<line class="svg-grid" x1="' + xPosition.get(value) + '" y1="' + top + '" x2="' + xPosition.get(value) + '" y2="' + (height - bottom) + '"/><text class="svg-label" x="' + xPosition.get(value) + '" y="' + (height - 23) + '" text-anchor="middle">' + escapeSvg(value) + "</text>").join("");
  const horizontalGrid = yTicks.map((value) => '<line class="svg-grid" x1="' + left + '" y1="' + y(value) + '" x2="' + (width - right) + '" y2="' + y(value) + '"/><text class="svg-label" x="' + (left - 8) + '" y="' + (y(value) + 3) + '" text-anchor="end">' + formatDecimal(value, value % 1 ? 1 : 0) + "×</text>").join("");
  const baseline = '<line class="scaling-baseline" x1="' + left + '" y1="' + y(1) + '" x2="' + (width - right) + '" y2="' + y(1) + '"/><text class="scaling-baseline-label" x="' + (width - right) + '" y="' + (y(1) - 6) + '" text-anchor="end">1× series baseline</text>';
  const marks = series.map((item) => {
    const path = item.points.map((point, index) => (index ? "L" : "M") + xPosition.get(point.x) + "," + y(point.y)).join(" ");
    const dash = item.scalingType === "weak" ? ' stroke-dasharray="8 5"' : "";
    const label = scalingSeriesAccessibleLabel(item);
    const points = item.points.map((point) => '<circle class="scaling-series-point" ' + conditionTargetAttributes(point) + ' cx="' + xPosition.get(point.x) + '" cy="' + y(point.y) + '" r="5"><title>' + escapeSvg(label + "; " + point.label) + "</title></circle>").join("");
    return '<g class="scaling-series" style="color:' + item.color + '"><path class="scaling-series-line" d="' + path + '"' + dash + "/><title>" + escapeSvg(label) + "</title>" + points + "</g>";
  }).join("");
  const labels = '<text class="svg-label svg-axis-title" x="' + width / 2 + '" y="' + (height - 4) + '" text-anchor="middle">Allocated GPUs</text><text class="svg-label svg-axis-title" transform="translate(15 ' + height / 2 + ') rotate(-90)" text-anchor="middle">Speedup relative to series baseline</text>';
  const svg = svgElement(width, height, verticalGrid + horizontalGrid + baseline + marks + labels);
  svg.classList.add("combined-scaling-svg");
  svg.setAttribute("aria-label", series.length + " isolated scaling series plotted together by GPU count and relative speedup");
  return svg;
}

function scalingSeriesAccessibleLabel(series) {
  return [series.platform, series.comparability, series.workload, series.strategy, series.precision, series.scalingType + " scaling"].join(" · ");
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
  const xTicks = pointAxisTicks(series.points.map((point) => point.x), xDomain);
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
  const workers = buildDataMovementSeries(result.rows, "num_workers");
  const prefetch = buildDataMovementSeries(result.rows, "prefetch_factor");
  const pinned = buildPinnedMemoryGroups(result.rows);
  renderLineCharts("movementWorkerCharts", workers, { empty: result.message || "No completed worker-count evidence.", xLabel: "DataLoader workers", yLabel: "loading time (ms)" });
  renderPinnedMemoryCharts("movementPinnedCharts", pinned, result.message || "No completed pinned-memory evidence.");
  renderLineCharts("movementPrefetchCharts", prefetch, { empty: result.message || "No completed prefetch-factor evidence.", xLabel: "prefetch factor", yLabel: "loading time (ms)" });
  document.getElementById("movementWorkersStatus").textContent = result.message || workers.length + " controlled curves";
  document.getElementById("movementPinnedStatus").textContent = result.message || pinned.length + " controlled comparisons";
  document.getElementById("movementPrefetchStatus").textContent = result.message || prefetch.length + " controlled curves";
  const allWorkers = buildDataMovementSeries(state.dataMovementSummary, "num_workers");
  const allPrefetch = buildDataMovementSeries(state.dataMovementSummary, "prefetch_factor");
  const allPinned = buildPinnedMemoryGroups(state.dataMovementSummary);
  markPanel("movementWorkerCharts", workers.length > 0, allWorkers.length > 0, "DataLoader worker sweep", "data_movement_summary.csv", true);
  markPanel("movementPinnedCharts", pinned.length > 0, allPinned.length > 0, "Pinned-memory effect", "data_movement_summary.csv", true);
  markPanel("movementPrefetchCharts", prefetch.length > 0, allPrefetch.length > 0, "Prefetch-depth sweep", "data_movement_summary.csv", true);
}

function buildDataMovementSeries(rows, studyType) {
  const definitions = {
    num_workers: {
      xField: "num_workers",
      xLabel: "DataLoader workers",
      title: "Worker count",
      constants: ["platform_id", "comparability", "workload", "throughput_unit", "pinned_memory", "prefetch_factor"],
    },
    prefetch_factor: {
      xField: "prefetch_factor",
      xLabel: "Prefetch factor",
      title: "Prefetch depth",
      constants: ["platform_id", "comparability", "workload", "throughput_unit", "num_workers", "pinned_memory"],
    },
  };
  const definition = definitions[studyType];
  if (!definition) return [];
  const eligible = rows.filter((row) => row.study_type === studyType && number(row[definition.xField]) >= 0 && number(row.avg_data_loading_seconds) > 0);
  const grouped = groupRows(eligible, (row) => definition.constants.map((field) => row[field] || "unspecified"));
  return Object.values(grouped).map((group) => {
    const first = group[0];
    return {
      title: first.platform_id + " · " + first.comparability + " · " + first.workload,
      subtitle: definition.title + " · loading and transfer time in milliseconds",
      platform: first.platform_id,
      comparability: first.comparability,
      workload: first.workload,
      unit: first.throughput_unit,
      studyType,
      comparison: comparisonContext(group, [definition.xLabel], definition.constants),
      points: group.map((row) => ({
        x: number(row[definition.xField]),
        y: number(row.avg_data_loading_seconds) * 1000,
        label: definition.xLabel + " " + row[definition.xField] + ": loading " + formatDecimal(number(row.avg_data_loading_seconds) * 1000, 2) + " ms; transfer " + formatDecimal(number(row.avg_transfer_seconds) * 1000, 2) + " ms; " + formatInteger(number(row.avg_throughput_tokens_sec)) + " " + row.throughput_unit + "; " + row.trials + " trials",
        conditionId: conditionId(row),
      })).sort((left, right) => left.x - right.x),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function buildPinnedMemoryGroups(rows) {
  const eligible = rows.filter((row) => row.study_type === "pinned_memory" && number(row.avg_data_loading_seconds) > 0);
  const grouped = groupRows(eligible, (row) => [row.platform_id || "unlabelled", row.comparability || "unspecified", row.workload || "unassigned", row.throughput_unit || "unit_not_recorded", row.num_workers || "unspecified", row.prefetch_factor || "unspecified"]);
  return Object.values(grouped).map((group) => {
    const first = group[0];
    return {
      title: first.platform_id + " · " + first.comparability + " · " + first.workload,
      subtitle: "Pinned-memory comparison · loading and transfer time in milliseconds",
      platform: first.platform_id,
      comparability: first.comparability,
      workload: first.workload,
      unit: first.throughput_unit,
      comparison: comparisonContext(group, ["Pinned memory"], ["platform_id", "comparability", "workload", "throughput_unit", "num_workers", "prefetch_factor"]),
      entries: group.map((row) => ({
        label: booleanLabel(row.pinned_memory, "Pinned memory on", "Pinned memory off"),
        value: number(row.avg_data_loading_seconds) * 1000,
        transfer: number(row.avg_transfer_seconds) * 1000,
        throughput: number(row.avg_throughput_tokens_sec),
        trials: number(row.trials),
        unit: row.throughput_unit,
        conditionId: conditionId(row),
      })).sort((left, right) => left.label.localeCompare(right.label)),
    };
  }).sort((left, right) => left.title.localeCompare(right.title));
}

function renderPinnedMemoryCharts(containerId, groups, emptyMessage) {
  const container = document.getElementById(containerId);
  container.replaceChildren();
  if (!groups.length) {
    container.append(emptyState(emptyMessage));
    return;
  }
  groups.forEach((group) => container.append(pinnedMemoryChartCard(group)));
}

function pinnedMemoryChartCard(group) {
  const card = chartCard(group.title, group.subtitle, group.comparison);
  const off = group.entries.find((entry) => entry.label.endsWith("off"));
  const on = group.entries.find((entry) => entry.label.endsWith("on"));
  if (off && on && off.value > 0) {
    const loadingChange = ((on.value - off.value) / off.value) * 100;
    const throughputChange = off.throughput > 0 ? ((on.throughput - off.throughput) / off.throughput) * 100 : 0;
    const effect = document.createElement("div");
    effect.className = "movement-effect";
    effect.innerHTML = '<strong>' + signedPercent(loadingChange) + ' loading time</strong><span>' + signedPercent(throughputChange) + " training throughput with pinned memory</span>";
    card.append(effect);
  }
  const bars = document.createElement("div");
  bars.className = "movement-comparison-bars";
  const top = max(group.entries.map((entry) => entry.value)) || 1;
  group.entries.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "bar-row movement-comparison-row";
    setConditionTarget(row, entry.conditionId);
    const stateClass = entry.label.endsWith("on") ? "checkpointing-on" : "checkpointing-off";
    row.innerHTML = '<div class="bar-meta"><strong>' + escapeHtml(entry.label) + '</strong><span>' + formatDecimal(entry.value, 2) + ' ms loading</span></div>' +
      '<div class="bar-detail">' + formatDecimal(entry.transfer, 2) + " ms transfer · " + formatInteger(entry.throughput) + " " + escapeHtml(entry.unit) + " · " + formatInteger(entry.trials) + ' trials</div>' +
      '<div class="bar-track"><div class="bar-fill ' + stateClass + '" style="width: ' + Math.max(3, (entry.value / top) * 100) + '%"></div></div>';
    bars.append(row);
  });
  card.append(bars);
  return card;
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
  safeRender();
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
    return values.length === 1 ? [{ label: displayFieldLabel(key), value: displayContextValue(key, values[0]) }] : [];
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
  const labels = fields.filter((key) => new Set(rows.map((row) => String(row[key] || "").trim()).filter(Boolean)).size > 1).map(displayFieldLabel);
  return labels.length ? labels : ["Configuration being tested"];
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

function median(values) {
  const numeric = values.filter((value) => Number.isFinite(value)).sort((left, right) => left - right);
  if (!numeric.length) return 0;
  const middle = Math.floor(numeric.length / 2);
  return numeric.length % 2 ? numeric[middle] : (numeric[middle - 1] + numeric[middle]) / 2;
}

function displayName(key) {
  return key.replace(/([a-z0-9])([A-Z])/g, "$1 $2").replaceAll("_", " ").toLowerCase();
}

function displayFieldLabel(key) {
  return {
    platform: "Platform",
    platform_id: "Platform",
    comparability: "Configuration class",
    workload: "Workload",
    throughput_unit: "Throughput measure",
    precision: "Precision",
    strategy: "Strategy",
    gpus: "GPU count",
    global_batch_size: "Global batch size",
    per_gpu_batch_size: "Per-GPU batch size",
    parameter_sequence_length: "Sequence length",
    mode: "Mode",
  }[key] || displayName(key).replace(/^./, (character) => character.toUpperCase());
}

function displayContextValue(key, value) {
  if (key === "platform" || key === "platform_id") return displayPlatform(value);
  if (key === "comparability") return displayComparability(value);
  if (key === "workload") return displayWorkload(value);
  if (key === "throughput_unit") return humanizeThroughputUnit(value);
  if (key === "strategy") return displayStrategy(value);
  if (key === "precision") return String(value || "").toUpperCase();
  return String(value || "").replaceAll("_", " ");
}

function displayPlatform(value) {
  return { h100: "H100", l40s: "L40S" }[String(value || "").toLowerCase()] || String(value || "Unlabelled platform").toUpperCase();
}

function displayComparability(value) {
  return {
    exact: "Original configuration",
    capacity_adjusted: "Capacity-adjusted",
    platform_specific: "Platform-specific",
  }[value] || String(value || "Unspecified configuration").replaceAll("_", " ");
}

function displayWorkload(value) {
  return {
    distilbert: "DistilBERT",
    gpt2_small: "GPT-2 Small",
    tinyllama: "TinyLlama",
  }[value] || String(value || "Unassigned workload").replaceAll("_", " ");
}

function displayStrategy(value) {
  return {
    ddp: "DDP",
    fsdp: "FSDP",
    deepspeed_zero1: "DeepSpeed ZeRO-1",
    deepspeed_zero2: "DeepSpeed ZeRO-2",
    deepspeed_zero3: "DeepSpeed ZeRO-3",
    none: "Single GPU",
  }[value] || String(value || "Unspecified strategy").replaceAll("_", " ");
}

function humanizeThroughputUnit(value) {
  return {
    training_tokens_per_second: "Training tokens per second",
    generated_tokens_per_second: "Generated tokens per second",
    input_tokens_per_second: "Input tokens per second",
  }[value] || String(value || "Throughput unit not recorded").replaceAll("_", " ");
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

function signedPercent(value) {
  const sign = value > 0 ? "+" : "";
  return sign + formatDecimal(value, 1) + "%";
}

function formatCompact(value) {
  return Number(value || 0).toLocaleString(undefined, { notation: "compact", maximumFractionDigits: 2 });
}

function pointAxisTicks(values, domain) {
  const ticks = [...new Set(values.filter((value) => Number.isFinite(value)))].sort((left, right) => left - right);
  if (ticks.length && ticks.length <= 8) return ticks;
  return [domain[0], (domain[0] + domain[1]) / 2, domain[1]];
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
    buildEnergyEfficiencyGroups,
    buildEnergyPowerEntries,
    buildEnergyPowerScalingSeries,
    buildEnergyToSolutionGroups,
    buildExperimentDetail,
    filterScalingSeries,
    buildGradientAccumulationSeries,
    buildBroadHardwareComparisonGroups,
    buildHardwareComparisonGroups,
    buildHardwareLandscape,
    buildInferenceGroups,
    buildMemorySeries,
    buildDataMovementSeries,
    buildDecisionInsights,
    buildPhaseDistributionEntries,
    buildPinnedMemoryGroups,
    buildRq1ScalingSeries,
    buildRq2StrategyGroups,
    buildResearchQuestionOverview,
    buildScalingSeries,
    buildStrategyComparisonGroups,
    buildStudyCoverageEntries,
    buildPrecisionComparisonGroups,
    buildUncertaintyGroups,
    comparisonContext,
    dashboardAlertMessage,
    parseCsv,
    pointAxisTicks,
    precisionColor,
    scalingColors,
    strategyColor,
    summarizePhases,
  };
}
