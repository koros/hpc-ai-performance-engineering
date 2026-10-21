#!/usr/bin/env Rscript
# Run from the repository root. Uses base R only; no package installation needed.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[[1]] else "results/figures/r_study_review"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
qwen_source <- "results/plot_data/inference_concurrency/qwen7b_long_prompt_conditions.csv"
scaling_source <- "results/plot_data/training_scaling/scaling_efficiency_curves.csv"
q <- read.csv(qwen_source, stringsAsFactors = FALSE)
q <- subset(q, platform == "l40s" & prompt_tokens == 1280 &
              output_tokens_requested == 128 & serving_layout %in% c("tp2_dp4", "tp4_dp2"))
s <- read.csv(scaling_source, stringsAsFactors = FALSE)
s <- subset(s, platform_id == "l40s" & experiment_id %in% paste0("E", 386:433))
# Fail rather than silently pool runtime cohorts or plot incomplete comparisons.
stopifnot(nrow(q) == 8, all(q$trial_count == 3), length(unique(q$runtime_cohort_id)) == 1,
          nrow(s) == 48, all(s$completed_trials == 3),
          length(unique(s$runtime_cohort_id)) == 1,
          all(s$scaling_basis == "matched_per_gpu_workload"))
stopifnot(!anyDuplicated(q[c("serving_layout", "concurrency")]),
          !anyDuplicated(s[c("workload", "gpus", "gradient_accumulation_steps")]))
colors <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7")
layouts <- c("tp2_dp4", "tp4_dp2")
labels <- c("TP2 x 4 replicas", "TP4 x 2 replicas")
render <- function(name, draw, width = 9, height = 6) {
  for (extension in c("png", "pdf")) {
    path <- file.path(out, paste0(name, ".", extension))
    if (extension == "png") png(path, width = width, height = height, units = "in", res = 180)
    else pdf(path, width = width, height = height, useDingbats = FALSE)
    tryCatch(draw(), finally = dev.off())
    message("Saved ", path)
  }
}
qwen_plot <- function(metric, ylabel, title) {
  par(mar = c(5, 5, 4, 1), oma = c(3, 0, 0, 0), las = 1, family = "sans")
  lo <- q[[paste0(metric, "_ci95_low")]]
  hi <- q[[paste0(metric, "_ci95_high")]]
  stopifnot(all(is.finite(c(lo, hi))))
  plot(NA, xlim = c(0.65, 4.35), ylim = c(0, max(hi) * 1.22), xaxt = "n",
       xlab = "Concurrent requests", ylab = ylabel, main = title)
  axis(1, 1:4, c("1", "2*", "4", "8"))
  abline(h = axTicks(2), col = "grey92", lty = 1)
  for (i in seq_along(layouts)) {
    d <- q[q$serving_layout == layouts[i], ]
    x <- match(d$concurrency, c(1, 2, 4, 8)) + c(-0.065, 0.065)[i]
    arrows(x, d[[paste0(metric, "_ci95_low")]], x, d[[paste0(metric, "_ci95_high")]],
           angle = 90, code = 3, length = 0.05, col = colors[i])
    points(x, d[[paste0(metric, "_mean")]], pch = c(16, 17)[i], col = colors[i], cex = 1.25)
  }
  legend("topright", labels, col = colors[1:2], pch = c(16, 17), bty = "n")
  mtext("L40S: 8 GPUs; Qwen2.5-7B; 1,280 prompt / 128 output tokens", side = 3, line = 0.3, cex = 0.8)
  mtext("Means and approximate 95% CIs across 3 independent trials per condition.",
        side = 1, outer = TRUE, line = 0.7, cex = 0.8)
  mtext("*Concurrency 2: follow-up motivated by observed results. Points do not locate an exact crossover.",
        side = 1, outer = TRUE, line = 1.8, cex = 0.75)
}
render("qwen7b_energy_by_concurrency", function() qwen_plot(
  "joules_per_output_token", "Measured GPU energy (J / output token)", "Energy ranking changes with concurrency"))
render("qwen7b_throughput_by_concurrency", function() qwen_plot(
  "throughput_tokens_sec", "Output throughput (tokens / second)", "Throughput of two eight-GPU layouts"))
render("qwen7b_first_token_latency", function() qwen_plot(
  "ttft_p95_ms", "Trial p95 time to first token (ms)", "First-token latency reveals a layout tradeoff"))
render("training_scaling_efficiency", function() {
  par(mfrow = c(1, 3), mar = c(4.5, 4.2, 3, 1), oma = c(4.5, 0, 3, 0), las = 1)
  workloads <- c(gpt2_small = "GPT-2 Small", distilbert = "DistilBERT", tinyllama = "TinyLlama")
  for (w in names(workloads)) {
    d <- s[s$workload == w, ]
    plot(NA, xlim = c(0.8, 4.2), ylim = c(45, 105), xaxt = "n",
         xlab = "GPUs", ylab = "Scaling efficiency (%)", main = workloads[[w]])
    axis(1, 1:4, c(1, 2, 4, 8))
    abline(h = c(60, 80), col = "grey92")
    abline(h = 100, col = "grey55", lty = 2)
    for (i in 1:4) {
      z <- d[d$gradient_accumulation_steps == c(1, 2, 4, 8)[i], ]
      z <- z[order(z$gpus), ]
      x <- match(z$gpus, c(1, 2, 4, 8))
      lines(x, 100 * z$scaling_efficiency, col = colors[i], lwd = 1.5)
      arrows(x, 100 * z$efficiency_ci95_low, x, 100 * z$efficiency_ci95_high,
             angle = 90, code = 3, length = 0.025, col = colors[i])
      points(x, 100 * z$scaling_efficiency, col = colors[i], pch = 14 + i)
    }
    if (w == "gpt2_small") legend("bottomleft", legend = paste("Accumulation", c(1, 2, 4, 8)),
      col = colors, pch = 15:18, lty = 1, bty = "n", cex = 0.85)
  }
  mtext("L40S training: matched per-GPU workload scaling", outer = TRUE, side = 3, line = 1, font = 2, cex = 1.15)
  mtext("3 trials per condition; bars: exported approximate 95% CIs; dashed line: ideal efficiency.",
        side = 1, outer = TRUE, line = 1.8, cex = 0.8)
  mtext("Per-GPU batch and sequence fixed within each curve; global batch grows with GPU count and accumulation.",
        side = 1, outer = TRUE, line = 2.8, cex = 0.75)
}, width = 12, height = 5.5)
write.csv(q, file.path(out, "qwen7b_plotted_conditions.csv"), row.names = FALSE, na = "")
write.csv(s, file.path(out, "training_plotted_conditions.csv"), row.names = FALSE, na = "")
writeLines(c(capture.output(sessionInfo()), "", "Input MD5 checksums:",
             capture.output(tools::md5sum(c(qwen_source, scaling_source))),
             "", paste("Qwen runtime:", unique(q$runtime_cohort_id)),
             paste("Training runtime:", unique(s$runtime_cohort_id))), file.path(out, "plot_provenance.txt"))
