# AMIP Baseline Workflow

This folder contains runnable scripts for generating training data and training a first Lux model for the AMIP baseline experiment.

Generated data is stored once, at the package level, under `data/processed/`.

## Quick Start (REPL Preferred)

From this directory:

```bash
cd /home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline
```

### 1. Bootstrap dataset + train model

Preferred REPL workflow:

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/bootstrap_train.jl")
```

Terminal fallback:

```bash
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/bootstrap_train.jl")' 2>&1 | tee agent.log
```

What it does:
- Activates the repository root environment.
- Builds GoogleLES Arrow data over multiple site/month pairs until enough rows are collected.
- Scans full LES cases by default (`max_timesteps=0` means no timestep cap).
- Skips cfSites by default (safe for non-Caltech HPC machines).
- Trains a Lux model once enough samples are available.

### 2. Batch data generation

Use this when you want to collect a large corpus of cases in chunks:

Preferred REPL workflow (recommended to avoid repeated compilation):

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/build_data.jl")
run_preferred_batch_generate!()
```

Terminal fallback:

```bash
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/build_data.jl")'
```

Default behavior:
- Scans GoogleLES site IDs `0:499` and months `1, 4, 7, 10`.
- Processes up to `BATCH_SIZE=100` cases per batch.
- Forces full-case GoogleLES cache loading by default.
- Runs batch execution in serial mode by default (more stable for network-bound runs).
- Continues through all requested cases by default.
- Optional pause between batches can be enabled.

### 3. Performance profiling

Use this to get stage-by-stage time and allocation metrics before optimization:

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")
```

Optional profiling knobs:

```bash
SITE_ID=343 MONTH=1 PROFILE_TIMESTEPS=8 PROFILE_REPEATS=5 julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")'
```

See `PERF_PLAN.md` for optimization order and regression workflow.

### 4. Full training experiment (separate train/plot)

Implementation plan:
- Stage A: Train on all available processed GoogleLES data with train/test split.
- Stage B: Save model + run artifact (predictions, truths, losses, metrics).
- Stage C: Regenerate rich diagnostics from the saved artifact without retraining.

Train full experiment:

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/full_train_experiment.jl")
run_full_training!()
```

Terminal fallback:

```bash
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/full_train_experiment.jl"); run_full_training!()'
```

Regenerate full diagnostics later (no retraining):

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/plot_full_training_diagnostics.jl")
plot_full_training_diagnostics!()
```

Terminal fallback:

```bash
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/plot_full_training_diagnostics.jl"); plot_full_training_diagnostics!()'
```

Distributed mode is also supported if you launch Julia with workers, for example `julia -p 4 --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/build_data.jl")'` and set `PARALLEL_BACKEND=distributed`.
If `OhMyThreads` is present in the environment, you can use `PARALLEL_BACKEND=omt`.

Useful environment variables:
- `CANDIDATE_SITES="10,11,12"`
- `CANDIDATE_MONTHS="1,7"`
- `BATCH_SIZE=100`
- `PARALLEL_BACKEND=threads|serial|distributed|omt`
- `PAUSE_BETWEEN_BATCHES=true|false`
- `FORCE_REPROCESS=true|false`
- `MAX_TIMESTEPS=0` for full-case scans
- `MLCD_COARSENING_MODE=hybrid|block|sliding` (default `hybrid` if unset). Legacy values `binary`, `convolutional`, `conv`, etc. are accepted with a **one-time warning** and treated as `hybrid`; unset any old `export MLCD_COARSENING_MODE=convolutional` in your shell init once you see the warning.
- `MLCD_SLIDING_OUTPUTS_H`, `MLCD_SLIDING_OUTPUTS_V`, `MLCD_SLIDING_OUTPUTS_Z` (default `2` each), `MLCD_SLIDING_WINDOW_BUDGET_H` — see `tabular_build_options_from_env` in `utils/build_training_common.jl`.

## Script Inventory

- `generate_data.jl`
  - Function: `generate_data!(; ...)`
  - Purpose: Generate Arrow files for one site/month/experiment configuration.
  - Graceful behavior: If cfSites fields are missing locally, it logs a skip message and continues.

- `build_data.jl`
  - Batch data-generation wrapper with chunked parallel execution.

- `train.jl`
  - Runs Lux training on `data/processed` using `utils/train_lux.jl`.

- `bootstrap_train.jl`
  - Robust workflow for this machine.
  - Repeats data generation over candidate site/month pairs until `min_rows` is reached, then trains.

- `full_train_experiment.jl`
  - Full training run with train/test split and expanded target set (`q_*`, `var_*`, `cov_*`).
  - Saves `lux_full_model.jls` and `lux_full_run_artifact.jls`.

- `plot_full_training_diagnostics.jl`
  - Plot-only diagnostics from `lux_full_run_artifact.jls`.
  - Produces richer target distributions/scatter and metrics table without retraining.

- `profile_googleles_pipeline.jl`
  - Stage-by-stage profiler for metadata load, cache load, and in-memory timestep processing.

- `PERF_PLAN.md`
  - Optimization and regression plan for performance work.

## Output Locations

- Experiment data root:
  - `data/processed`

- Smoke test data root:
  - `data/tests/amip_baseline`

- Trained model artifact:
  - `data/models/lux_model.jls`

## Current Full-Training Behavior

The full experiment uses [utils/train_lux.jl](../../utils/train_lux.jl) with:

- standardized inputs and targets,
- two-head outputs for phase targets (`q_liq`, `q_ice`):
  - magnitude channel,
  - presence logit channel,
- combined phase loss (presence BCE + conditional magnitude MSE),
- validation early stopping with best-checkpoint restore.

Important knobs when calling `run_full_training!`:

- `epochs`: high ceiling is acceptable because early stopping controls actual stop.
- `early_stopping_patience`: epochs without meaningful validation improvement.
- `early_stopping_min_delta`: minimum validation improvement required to reset patience.

Important knobs passed through `train_model`:

- `phase_presence_threshold`: defines nonzero target for presence labels.
- `phase_presence_loss_weight`: weight of presence BCE term.
- `phase_magnitude_loss_weight`: weight of magnitude MSE term.
- `phase_presence_prob_threshold`: decode threshold for exact zero vs nonzero emission.

## Common Issues

- No rows after generation:
  - Cause: all cloudy samples masked out for selected timesteps/site.
  - Fix: use `bootstrap_train.jl` (it expands sites/months and timesteps).

- cfSites path missing:
  - Expected on non-HPC systems.
  - The workflow should log a skip and proceed with GoogleLES.

## Notes

- All scripts here are intended to be normal CLI-runnable scripts.
- Prefer the bootstrap workflow when the goal is obtaining a trained model, not just a smoke test.
