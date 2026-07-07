# Config Templates

These templates are valid `ExperimentConfig` JSON files that can be copied or generated into concrete experiment configs.

Conventions:

- `experiment_id` starts with `template_` for reusable templates and `E###` for matrix runs.
- `metadata.matrix_id` links a concrete config to `docs/experiment_matrix.md`.
- `metadata.template` records the template family used to create the config.
- `runtime.output_dir` should remain `results/raw` unless a phase needs a separate raw-output staging path.
- Environment-validation configs use `runtime.precision = "bf16"` when the matrix precision is `N/A`; the original matrix value is preserved in `metadata.matrix_precision_config`.

Recommended concrete config path:

```text
configs/<phase>/<experiment_id>_<short_description>.json
```
