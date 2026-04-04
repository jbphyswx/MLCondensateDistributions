# GoogleLES → Arrow pipeline (`build_tabular`)

This document is the **canonical description** of how GoogleLES cases are turned into per-case `.arrow` files. It exists so the design (network-minimal z loading, local materialization, and vertical-mask semantics) stays unambiguous when the code evolves.

**Zarr storage axes:** Which Julia index is z vs t, and how `_ARRAY_DIMENSIONS` relates to `size`/`chunks`, is specified in [googleles_zarr_layout.md](googleles_zarr_layout.md). Any change to loaders or chunk-aware logic should follow that note.

## Source of truth (what the package actually loads)

The Julia package **`MLCondensateDistributions` includes only**:

| File | Role |
|------|------|
| `utils/build_training_data.jl` | `GoogleLES.build_tabular` orchestration (timesteps, masks, span loop, Arrow write) |
| `utils/build_training_common.jl` | Zarr slicing helpers, `_load_googleles_timestep_fields_into_span_list!`, shared constants |
| `utils/GoogleLES.jl` | `load_zarr_simulation`, physics helpers (e.g. condensate partition) |

```36:37:src/MLCondensateDistributions.jl
include("../utils/build_training_common.jl")
include("../utils/build_training_data.jl")
```

**`utils/__deprecated__/` is not `include`d anywhere.** Nothing in the supported pipeline should import it. If you see references to that folder in old notes or local scripts, treat them as stale; the folder may be removed.

## End-to-end algorithm (`GoogleLES.build_tabular`)

1. **Open Zarr** via `GoogleLES.load_zarr_simulation` (HTTP store, consolidated metadata when available).

2. **`q_c` staging (full-case mode)**  
   When the working set policy selects full-case mode, all timesteps of `q_c` are loaded once into a 4D lazy/permuted view cache (`_load_googleles_cache`), then **each timestep** is copied into a preallocated 3D buffer (`q_c_buf .= q_cache["q_c"][local_t, :, :, :]`).  
   That copy **materializes** one timestep so cloud checks and masks never touch lazy remote elements repeatedly.

3. **Early skip**  
   `_has_cloud_after_2x2(q_c_buf)` uses only local memory. Timesteps with no cloud at the first horizontal coarsening scale are skipped before any other fields are fetched.

4. **Native vertical keep mask (`z_keep_mask`)**  
   - `empty_z_levels = identify_empty_z_levels(q_c_buf, threshold)` — per native `k`, is the entire `(x,y)` plane dry?  
   - `z_keep_mask = build_z_level_keep_mask(empty_z_levels, 1, z_future_factors)` where `z_future_factors` comes from **`compute_z_coarsening_scheme` on the full-column `dz_native_profile`** (same schedule the tabular builder will use for vertical reduction in principle).  
   So `true` entries mark the **most expansive set of native z-indices** we conservatively keep for downstream vertical coarsening logic, not “cloudy only” in a narrow sense.

5. **Logical (mask) spans**  
   `_collect_true_spans(z_keep_mask)` lists every contiguous run of `true` in native z.  
   These are the **only** z-indices that downstream physics uses (`process_abstract_chunk` is
   still invoked **once per logical span**, with `@view`s restricted to that span).

6. **Non-`q_c` materialization: overlapping storage z-chunks (default)**  
   Logical spans are mapped to storage z-chunk index intervals; spans whose intervals overlap (transitively) form **one group**. For each group, **`_load_googleles_timestep_fields_into_span_list!`** runs **one `_load_googleles_timestep_fields_into!` per original span**, each with **that span’s native `z_range` only** — e.g. `10:20` then `40:50`. The application **never** widens the Zarr request to a chunk hull like `1:60`.  

   `MLCD_GOOGLELES_Z_CHUNK_MERGE=0`: each group is a single span (same narrow slices, no batching by chunk overlap).

   Opt-in **full column** once per cloudy timestep: `MLCD_GOOGLELES_NONQC_STRATEGY=full_timestep`.

   Verbose logs report z-chunk size via `_googleles_effective_z_chunk_size`. See [googleles_zarr_layout.md](googleles_zarr_layout.md).

   All reads **`copyto!` into local buffers** before any compute (no lazy-remote math).

7. **Partition and tabular build (per logical span)**  
   One builder pass per mask span with `dz_native_profile[z_range]` matching that span only.

8. **Case-level table concat**  
   Non-empty chunk `DataFrame`s are accumulated with **`append!` into one case-level `DataFrame`** (via a `Ref`), then **`Arrow.write`** once — avoids `reduce(vcat, Vector{DataFrame})` and repeated column growth from naive `push!` rows.

## Design rules (easy to get wrong)

1. **Never compute on lazy Zarr-backed arrays** — always `copyto!` into preallocated buffers before arithmetic or coarsening. See [v2_remote_load_reduction_questions.md](v2_remote_load_reduction_questions.md) (“CRITICAL LESSON”).

2. **`scratch` dict**  
   One `Dict{String, AbstractArray{Float32,3}}` per case (or batch) loop is **only** for short-lived views between `empty!` and `copyto!`; it is not a second data cache.

3. **Do not widen mask spans to chunk hulls in Zarr calls**  
   Request **only** each logical span’s `z_range`. Internal chunk fetch/decode is Zarr’s responsibility.

## Per-timestep profiling (serial)

Set **`MLCD_GOOGLELES_TIMESTEP_PROFILE=1`** when calling `GoogleLES.build_tabular` to print running averages (over **cloudy** timesteps only) for: `qc` slab copy, `prep` (masks/spans), `nonqc_zarr`, `tabular` (`process_abstract_chunk` path). Optional **`MLCD_GOOGLELES_TIMESTEP_PROFILE_EACH=1`** prints every cloudy timestep. See `docs/ARROW_PIPELINE_PERF_PLAN.md`.

Regression tests: `test/test_googleles_timestep_profile.jl` (profile accumulator), `test/test_googleles_nonqc_strategy.jl` (`MLCD_GOOGLELES_NONQC_STRATEGY` parsing).

## Coarsening mode (`TabularBuildOptions` / `MLCD_COARSENING_MODE`)

Tabular rows are built via `DatasetBuilderImpl.process_abstract_chunk`, driven by `opts.coarsening_mode` (from `TabularBuildOptions` or `tabular_build_options_from_env()`):

| Mode | Meaning |
|------|--------|
| `hybrid` | Default: block ladder where appropriate + sliding valid-box samples for gap scales. |
| `block` | Truncated non-overlapping 3D blocks; optional explicit factor list via `spatial_info.block_triples` in library callers. |
| `sliding` | Sliding valid-box reductions only (see `sliding_outputs_*` / optional `sliding_stride_*` in `spatial_info`). |

Shell / batch scripts: set `MLCD_COARSENING_MODE` to `hybrid`, `block`, or `sliding`. Older env values such as `convolutional` or `binary` still parse but log a deprecation warning and behave as `hybrid`.

## Related docs

- [v2_remote_load_reduction_questions.md](v2_remote_load_reduction_questions.md) — remote Zarr behavior, benchmarks, and the lazy-view gridlock failure mode.
- Docstring on `GoogleLES.build_tabular` in `utils/build_training_data.jl` — kept in sync with this file when behavior changes.
