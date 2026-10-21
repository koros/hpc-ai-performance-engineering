# Run from the repository root after make export-plot-data PYTHON=python.
library(readr)
library(dplyr)
library(ggplot2)

conditions <- read_csv(
  "results/plot_data/inference_concurrency/qwen7b_long_prompt_conditions.csv",
  show_col_types = FALSE
)
plot_data <- conditions |>
  filter(platform == "l40s", serving_layout %in% c("tp2_dp4", "tp4_dp2"),
         prompt_tokens == 1280, output_tokens_requested == 128)
# Faceting prevents silently comparing records with different runtime identities.
p <- ggplot(plot_data, aes(concurrency, joules_per_output_token_mean,
                           colour = serving_layout)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = joules_per_output_token_ci95_low,
                   ymax = joules_per_output_token_ci95_high), width = 0.12) +
  facet_wrap(~runtime_cohort_id, labeller = labeller(runtime_cohort_id = function(x) substr(x, 1, 10))) +
  scale_x_continuous(breaks = c(1, 2, 4, 8)) +
  labs(x = "Concurrent requests", y = "Measured GPU energy / output token (J)",
       colour = "Eight-GPU layout",
       caption = paste("Points are condition means; bars are approximate trial-level 95% CIs.",
                       "Concurrency 2: follow-up motivated by observed results.")) +
  theme_classic()
print(p)
# Save explicitly when ready:
# ggsave("results/figures/qwen_energy_concurrency.pdf", p, width = 7, height = 4)
