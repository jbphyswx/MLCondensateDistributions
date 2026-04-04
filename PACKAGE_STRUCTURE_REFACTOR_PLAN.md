# PACKAGE STRUCTURE REFACTOR PLAN

Status: active
Owner: MLCondensateDistributions maintainers
Scope: entire repository (core package + experiments)

## 1. Purpose

This plan restructures the package to eliminate script-specific permanent APIs, reduce duplicated helper methods, and establish clear module boundaries for reuse.

Primary goals:
- Keep core package APIs generic and reusable.
- Keep visualization modules plotting-only.
- Keep experiment scripts orchestration-only.
- Consolidate duplicated helpers across AMIP and analysis scripts.
- Make future analyses (for example liquid_fraction_impact_covariance) require minimal new code.

## 2. Constraints and Non-Goals

- Do not change scientific results unless explicitly part of bug fixes.
- Do not break existing experiment entrypoints without compatibility shims.
- Refactor architecture first; larger algorithmic redesigns are out-of-scope here.

## 3. Target Package Layout

## 3.1 Core module tree (target)
- src/MLCondensateDistributions.jl
- src/models.jl
- utils/
  - paths.jl
  - workflow_state.jl
  - dynamics.jl
  - coarse_graining.jl
  - dataset_builder.jl
  - GoogleLES.jl
  - cfSites.jl
  - build_training_data.jl
  - dataloader.jl
  - data_handling.jl
  - analysis.jl (new)
  - io_helpers.jl (new, optional)
  - env_helpers.jl (new, optional)
  - train_lux.jl
- viz/
  - viz.jl (non-plotting namespace + output-path helpers)
  - ext/CairoMakieExt.jl (plotting methods only)

## 3.2 Responsibility boundaries (hard rules)
- DataHandling: file discovery, Arrow loading, generic column selection, generic table building.
- Analysis: grouped reductions, quantiles, pivots, 2D matrices for heatmaps.
- Viz: rendering only (no Arrow/file-discovery/business logic).
- Experiments: choose parameters, call package APIs, save outputs.

## 4. Current Audit (function-level)

Audit source: function inventory across src/utils/viz/experiments.

## 4.1 Keep in core modules (already appropriate)
- utils/coarse_graining.jl: coarse-graining and covariance kernels.
- utils/dataset_builder.jl: chunk processing and flattening.
- utils/build_training_data.jl: orchestrated data build.
- utils/train_lux.jl: training loop + artifact handling.
- utils/paths.jl, utils/workflow_state.jl, utils/GoogleLES.jl, utils/cfSites.jl.

## 4.2 Methods to consolidate/move to generic modules

### Data handling overlaps
- utils/data_handling.jl:
  - collect_arrow_files
  - select_columns_by_prefix
  - load_resolution_moments_df
- utils/dataloader.jl:
  - prune_incompatible_arrow_files!
  - load_processed_data
  - preview_processed_file

Actions:
- Replace `load_resolution_moments_df` with generic `load_arrow_columns` + optional schema checks.
- Keep `select_columns_by_prefix` as generic selector.
- Unify Arrow file discovery/validation in one place.
- Keep training-specific matrix conversion in dataloader, but use shared lower-level loaders.

### Visualization helpers currently duplicated in scripts
- experiments/amip_baseline/plot_full_training_diagnostics.jl has many private helpers (`_robust_limits`, `_plot_group_distributions`, `_plot_group_scatter`, `_plot_moment_heatmaps`, etc.)

Actions:
- Move generic plotting helpers into `ext/CairoMakieExt.jl`.
- Keep `viz/viz.jl` as the lightweight public namespace and path helper module.
- Keep script as thin runner: load artifact -> call Viz API.

### Analysis-specific math currently in Viz or scripts
- `Viz` plotting APIs currently live behind a file name that does not match the module boundary.

Actions:
- Keep non-rendering pieces in `utils/analysis.jl`.
- Move all CairoMakie-dependent plotting methods to `ext/CairoMakieExt.jl`.
- Keep `viz/viz.jl` as the public namespace and lightweight output-path helper layer.

## 4.3 Script-level helper duplication to remove

### experiments/amip_baseline/batch_generate_data.jl
Current local helper methods:
- parse_bool_env
- parse_int_list_env
- case_arrow_filename
- case_arrow_path
- count_rows
- case_rows
- case_is_satisfied
- expand_cases
- prompt_continue
- run_case!, run_case_batch, run_case_batch_omt, run_case_batch_distributed

Actions:
- Move generic env parsing to `utils/env_helpers.jl`.
- Reuse shared path/file helpers from DataHandling/Paths.
- Keep script-only orchestration methods that are truly workflow-specific.

### experiments/amip_baseline/bootstrap_train.jl and generate_data.jl
- Duplicated env parsing and row counting logic.

Actions:
- Replace duplicates with shared `env_helpers.jl` and shared data helpers.

### experiments/analyze_truth_data/scripts/resolution_impact_covariances.jl
- Should remain orchestration-only.

Actions:
- Keep only `run_resolution_impact_covariances!` and move reusable logic to package modules.

## 5. Concrete API Refactor Spec

## 5.1 DataHandling API (target)
- `list_arrow_files(data_dir; max_files=0)`
- `select_columns(colnames; names=Symbol[], prefixes=String[], regex=nothing)`
- `load_arrow_dataframe(data_dir; columns::Vector{Symbol}, max_files=0, drop_empty=true)`
- `validate_required_columns(df, required_cols)`
- `grouped_table(df, keys, target_cols; reducer=:median)` (or in Analysis module)

Compatibility:
- Keep `collect_arrow_files` as wrapper to `list_arrow_files` during migration.
- Keep `load_resolution_moments_df` as temporary wrapper and mark deprecated.

## 5.2 Analysis API (new module)
- `group_reduce(df, by_cols, value_col; reducer=median)`
- `group_reduce_many(df, by_cols, value_cols; reducer=median)`
- `build_heatmap_matrix(df, x_col, y_col, value_col; reducer=median)`
- `quantile_summary(df, by_cols, value_col, qs=[0.1,0.5,0.9])`

## 5.3 Viz API (target)
- `plot_targets_vs_axis(grouped_tables_or_df, ...)`
- `plot_target_heatmaps(heatmap_inputs, ...)`
- `plot_training_diagnostics(artifact, out_dir; options...)`

Rule: Viz APIs accept prepared tables/matrices or artifact structs, not raw Arrow file paths.

## 6. Task Checklist (Execution)

## Phase A: Foundation
- [x] A1. Add `utils/analysis.jl` module and include/export it in `src/MLCondensateDistributions.jl`.
- [x] A2. Add `utils/env_helpers.jl` module for shared env parsing.
- [x] A3. Rename DataHandling API to generic names and add compatibility wrappers.
- [x] A4. Add module-level docs for DataHandling/Analysis/Viz responsibilities.

## Phase B: Migrate Existing Code
- [x] B1. Refactor `viz/resolution_analysis.jl` to consume Analysis results only.
- [x] B2. Refactor `experiments/analyze_truth_data/scripts/resolution_impact_covariances.jl` to orchestration-only.
- [x] B3. Migrate generic training diagnostic plot helpers from `experiments/amip_baseline/plot_full_training_diagnostics.jl` to `viz/training_diagnostics.jl`.
- [ ] B4. Replace duplicated env/file helper functions in AMIP scripts with shared helpers.

## Phase C: Cleanup and Deprecation
- [ ] C1. Mark transitional wrappers deprecated with clear replacement guidance.
- [ ] C2. Remove duplicated script-local helper methods after migration.
- [ ] C3. Remove hyperspecific names from permanent exports.

## Phase D: Tests
- [ ] D1. Add unit tests for generic DataHandling selectors/loaders.
- [ ] D2. Add unit tests for Analysis grouped reduction and heatmap matrix generation.
- [ ] D3. Add smoke tests for `resolution_impact_covariances` using small `max_files`.
- [ ] D4. Ensure existing AMIP tests still pass unchanged.

## Phase E: Docs and Developer Guidance
- [ ] E1. Update top-level README with architecture boundary diagram.
- [ ] E2. Update `utils/README.md` with API map and migration notes.
- [ ] E3. Add script authoring rules: no data loader logic in scripts; no plotting logic in DataHandling.

## 7. File-by-File Action Map

- src/MLCondensateDistributions.jl
  - Add includes/exports for Analysis and EnvHelpers.
  - Ensure exports include only reusable stable APIs.
  - Include `viz/viz.jl` as the public visualization namespace, with plotting implemented in the extension.

- utils/data_handling.jl
  - Convert specific loader to generic loader(s).
  - Keep temporary compatibility wrappers.

- utils/dataloader.jl
  - Use DataHandling lower-level loaders where possible.
  - Retain training matrix conversion here.

- viz/resolution_analysis.jl
  - Remove data prep logic and rely on Analysis/DataHandling inputs.

- experiments/analyze_truth_data/scripts/resolution_impact_covariances.jl
  - Keep as orchestration only.

- experiments/amip_baseline/plot_full_training_diagnostics.jl
  - Move reusable plotting helpers to `viz/training_diagnostics.jl`.

- experiments/amip_baseline/batch_generate_data.jl
- experiments/amip_baseline/bootstrap_train.jl
- experiments/amip_baseline/generate_data.jl
  - Replace duplicated parser/path helpers with shared modules.

## 8. Acceptance Criteria

- [ ] AC1. No script-specific helper is exported from package root.
- [ ] AC2. No Viz module function performs Arrow file discovery/loading.
- [ ] AC3. At least two different analysis scripts reuse the same generic DataHandling + Analysis APIs.
- [ ] AC4. Existing AMIP workflows run with unchanged user-facing commands.
- [ ] AC5. Tests cover new generic APIs and migration wrappers.

## 9. Progress Tracker

- [x] P1. Initial function inventory audit completed.
- [x] P2. Initial DataHandling vs Viz split started.
- [ ] P3. Generic API migration complete.
- [ ] P4. Script deduplication complete.
- [ ] P5. Deprecation cleanup complete.
- [ ] P6. Test and docs signoff complete.

## 10. Clean up Extraneous Files
- [x] C1. Removed the stray AMIP one-off scripts from `experiments/amip_baseline/`; permanent tests now live under `test/`.