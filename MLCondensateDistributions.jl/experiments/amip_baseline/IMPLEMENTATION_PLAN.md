# AMIP Baseline Implementation Plan

## Goals
- Make the generated Arrow data physically self-consistent.
- Add correctness tests for every transformation we own.
- Make diagnostics interpretable rather than visually misleading.
- Keep workflow bookkeeping visible and simple.
- Avoid silent data corruption or stale cached bookkeeping.

## 1. Correctness tests for transformations and calculations

### 1.1 Dataset builder invariants
- Add unit tests for `DatasetBuilder.process_abstract_chunk` that verify:
  - output schema is exactly what we expect;
  - all numeric outputs are finite;
  - `q_con == q_liq + q_ice` exactly or within a strict tolerance;
  - `tke >= 0` for all emitted rows;
  - covariance and variance outputs are consistent on synthetic data.
- Add a regression test that injects a non-finite source value and confirms the row is dropped before Arrow writing.

### 1.2 Coarse-graining correctness
- Add tests for `cg_2x2_horizontal` and `cg_2x_vertical` with hand-computed arrays.
- Add tests for `compute_covariance` using known analytic inputs.
- Add a test that verifies covariances stay finite when inputs are finite.

### 1.3 Physics-derived quantities
- Add a test that verifies TKE is derived from velocity variances, not raw signed velocity means.
- Add a test for condensate partitioning that verifies:
  - liquid fraction is clamped to `[0, 1]`;
  - warm inputs produce all-liquid condensate;
  - cold inputs produce all-ice condensate.
- Add a temperature reconstruction test if we later compute `ta` from stored thermodynamic state.

### 1.4 Loader and schema hygiene
- Add tests that `load_processed_data` rejects or prunes Arrow files with missing columns.
- Add tests that Arrow files with non-finite values are removed from the training set.
- Add tests that empty Arrow files do not get treated as valid training data.

## 2. Add `ta` and reprocess historical data

### 2.1 Decide storage strategy
- Add `ta` as an explicit QA column in the processed data schema, or add a dedicated derived-temperature field computed from stored thermodynamic state.
- Keep the training feature list stable unless `ta` is intentionally used as an input feature.
- Document the exact provenance of `ta` in `dataset_spec.md`.

### 2.2 Reprocess existing data
- Regenerate all Arrow case files after the schema update.
- Remove or quarantine any legacy Arrow files generated before the fix.
- Add a one-shot validation script that scans every Arrow file for:
  - finiteness;
  - exact schema compliance;
  - zero-row files;
  - balance identities such as `q_con = q_liq + q_ice`.

### 2.3 Backward compatibility
- Keep `load_processed_data` tolerant of extra QA columns if they exist.
- Do not silently use stale files if the schema version changes.

## 3. Workflow cache redesign

### 3.1 Make processed data self-describing
- Prefer case-local sentinel files or per-case metadata in the processed-data directory over an opaque shared cache table.
- If a case produced zero valid rows, store an explicit empty marker file rather than relying on a cache manifest alone.

### 3.2 Keep bookkeeping visible
- Make it obvious from the processed-data directory whether a case was:
  - not attempted;
  - attempted and empty;
  - attempted and written successfully;
  - attempted and rejected for correctness reasons.
- Keep the bookkeeping simple enough that it can be inspected without running code.

### 3.3 Retain workflow cache only if it adds value
- If `workflow_cache` remains, use it as an index, not as the source of truth.
- The Arrow files and per-case sentinels should remain the authoritative record.

## 4. Visualization improvements

### 4.1 Replace misleading plots
- Use robust axis limits and quantile-based scaling.
- Detect constant-truth targets and switch to a better visualization.
- Add conditional plots for `q_liq` and `q_ice` vs `qt` and `theta_li`.

### 4.2 Add output summaries
- Emit a `target_summary.csv` with per-target quantiles and zero fractions.
- Save diagnostics in a timestamped folder per training run.

## 5. Order of execution
1. Fix correctness bugs in the data pipeline.
2. Add tests for those bugs.
3. Regenerate processed data.
4. Update diagnostics plots.
5. Revisit whether `workflow_cache` should be replaced with per-case sentinels.

## 6. Acceptance criteria
- No negative TKE values in generated Arrow data.
- No non-finite numeric values written to Arrow.
- Correctness tests cover all owned transformations.
- Historical data can be regenerated cleanly after schema changes.
- Visualization outputs show physical relationships clearly enough to interpret.
