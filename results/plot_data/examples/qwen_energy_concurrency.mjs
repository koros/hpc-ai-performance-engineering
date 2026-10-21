// Serve results/plot_data/ over HTTP. Import this module in a page under examples/.
// Feed the returned traces/layout directly to Plotly.newPlot(element, traces, layout).
export async function qwenEnergyPlot() {
  const response = await fetch(new URL("../inference_concurrency/qwen7b_long_prompt_conditions.json", import.meta.url));
  if (!response.ok) throw new Error(`Dataset request failed: ${response.status}`);
  const rows = (await response.json()).filter(row => row.platform === "l40s" &&
    ["tp2_dp4", "tp4_dp2"].includes(row.serving_layout) &&
    row.prompt_tokens === 1280 && row.output_tokens_requested === 128);
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.serving_layout} · ${row.runtime_cohort_id.slice(0, 10)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const traces = [...groups].map(([name, values]) => {
    values.sort((a, b) => a.concurrency - b.concurrency);
    return {
      type: "scatter", mode: "markers", name,
      x: values.map(row => row.concurrency),
      y: values.map(row => row.joules_per_output_token_mean),
      error_y: { type: "data", symmetric: false,
        array: values.map(row => row.joules_per_output_token_ci95_high === null ? null :
          row.joules_per_output_token_ci95_high - row.joules_per_output_token_mean),
        arrayminus: values.map(row => row.joules_per_output_token_ci95_low === null ? null :
          row.joules_per_output_token_mean - row.joules_per_output_token_ci95_low) },
      text: values.map(row => `${row.experiment_id}; n=${row.trial_count}; ${row.study_design_note || "recorded comparison"}`)
    };
  });
  return { traces, layout: {
    xaxis: { title: "Concurrent requests", tickvals: [1, 2, 4, 8] },
    yaxis: { title: "Measured GPU energy / output token (J)" },
    title: "Qwen7B: TP2 versus TP4 at a fixed eight-GPU budget"
  }};
}
