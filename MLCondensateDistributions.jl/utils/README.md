# Utilities Reference

This directory contains the package implementation used by [src/MLCondensateDistributions.jl](../src/MLCondensateDistributions.jl).

## File Overview

- [paths.jl](paths.jl)
  - Canonical filesystem paths for processed data, models, and experiment outputs.

- [workflow_state.jl](workflow_state.jl)
  - Workflow bookkeeping and case-state helpers for data generation.

- [dynamics.jl](dynamics.jl)
  - Derived physical quantities (for example turbulent kinetic energy helpers).

- [coarse_graining.jl](coarse_graining.jl)
  - Horizontal/vertical coarsening operators and covariance utilities.

- [dataset_builder.jl](dataset_builder.jl)
  - Core transformation from high-resolution fields to coarse-grained tabular rows.
  - Applies sparsity filters and schema-ordered flattening.

- [GoogleLES.jl](GoogleLES.jl)
  - GoogleLES-specific metadata discovery and field loading.

- [cfSites.jl](cfSites.jl)
  - cfSites-specific pathing and field/stat accessors.

- [build_training_common.jl](build_training_common.jl)
  - Shared GoogleLES Zarr helpers (`_load_googleles_cache`, span materialization) and Arrow/case path helpers used by `build_training_data.jl`.

- [build_training_data.jl](build_training_data.jl)
  - Orchestrates case processing and writes Arrow outputs (`GoogleLES.build_tabular`, `cfSites.build_tabular`).

**Note:** `utils/__deprecated__/` is **not** included by `src/MLCondensateDistributions.jl` and is not part of the supported package API. Do not add `include`s pointing at it; it may be removed.

**Pipeline reference:** [docs/googleles_build_tabular.md](../docs/googleles_build_tabular.md) describes the GoogleLES → Arrow algorithm (z-mask, per-span loads, materialization rules).

**Zarr axes:** [docs/googleles_zarr_layout.md](../docs/googleles_zarr_layout.md) — how `size`/`chunks` line up with `_ARRAY_DIMENSIONS` in Julia (avoid indexing `chunks` by metadata list position).

- [dataloader.jl](dataloader.jl)
  - Reads Arrow files and returns feature/target matrices for training.

- [data_handling.jl](data_handling.jl)
  - Reusable Arrow/dataframe helpers shared by training and analysis.
  - Includes reusable resolution-oriented loaders used by analysis scripts.

- [train_lux.jl](train_lux.jl)
  - Lux training loop, early stopping, metric reporting, and artifact save.

## Design Notes

- Coarse-graining and dataset-building are source-agnostic once fields are loaded.
- Source adapters (`GoogleLES.jl`, `cfSites.jl`) provide loader-specific behavior.
- Training and diagnostics are decoupled:
  - training writes model + run artifact,
  - plotting scripts regenerate diagnostics from saved artifacts.
- Data-loading logic is centralized in `DataHandling`; visualization modules should
  consume those helpers instead of re-implementing file loaders.

## Runtime Notes

- For long AMIP runs, prefer REPL-based workflows to avoid repeated compilation.
- Progress logging can be controlled through script kwargs/environment settings in experiment scripts.
