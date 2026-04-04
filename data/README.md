# Generated Data Layout

Generated datasets and model artifacts for `MLCondensateDistributions` live here.

## Layout

- `data/processed/`
  - Shared Arrow datasets for all workflows.
- `data/tests/<name>/`
  - Temporary datasets created by smoke/integration tests.
- `data/models/`
  - Serialized model artifacts such as Lux checkpoints.
  - Typical full-run artifacts:
    - `lux_full_model.jls`
    - `lux_full_run_artifact.jls`
  - Diagnostics outputs:
    - `diagnostics_full/latest/` (default overwrite target)
    - optionally timestamped subdirectories when enabled in plotting script.

## Notes

- This directory is for generated output, not source code.
- Empty Arrow outputs should not be written; workflows should skip zero-row chunks.
- Generated files are ignored by git via the repository `.gitignore`.
- The run artifact is intended to be plot-replayable so diagnostics can be regenerated without retraining.
