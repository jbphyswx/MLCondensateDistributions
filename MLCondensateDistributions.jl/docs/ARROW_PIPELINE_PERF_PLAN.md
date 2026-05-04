# Arrow pipeline performance plan

Track serial-throughput work, your constraints, and implementation status. Update checkboxes as you go.

## Your constraints (from discussion)

| Topic | Your position | Implication |
|--------|----------------|-------------|
| Parallelism (cases / timesteps / threads) | Will add **after** serial is fully optimized | Defer multi-worker / threaded coarsening until this list is thinner |
| Local mirror of ~200 TB | **Not viable**; download-delete not attractive | Serial optimizations must assume **remote Zarr** stays hot |
| Fewer emitted stages / rows | Question: “still more than binary reduction?” | **Different axis**: binary reduction = how we *pool*; fewer rows = *what* we emit (e.g. conv/patch). Conv output can be fewer rows **or** different layout — not a strict superset of “binary ladder” |
| Profiler-first | OK to run | Use `experiments/amip_baseline/profile/profile_dataset_builder_breakdown.jl` (or existing `profile_v2_builder.jl`) on representative `NX,NY,NZ` |

## Vertical ladder vs remote reads (clarification)

- **Inside one `process_abstract_chunk` call**, data is already **in-memory** 3D slabs (or views). The **vertical ladder** recomputes coarser `z` by **local** `coarsen3d_vertical_mean` etc. There are **no extra Zarr reads per vertical stage**.
- **Per horizontal level**, you **do** run a **new** vertical schedule on **new** `(nx, ny)` after horizontal pooling — that’s **by design** for the statistics, not accidental double-fetch from remote.
- **Across native-`z` spans** in `build_tabular`, you may call `process_abstract_chunk` **multiple times** per timestep (one per mask span) → **repeated CPU** on the same `(x,y)` footprint, **not** repeated read of the same Zarr bytes if buffers are filled once per span/group.

---

## Section 3 issues — status

### 3a — Redundant `compute_z_coarsening_scheme` every horizontal level

- **Status:** **Done** — `z_schemes` / `z_scheme_factors` computed once per chunk in `utils/dataset_builder_impl.jl`.
- **Also:** `future_z_factors` uses `@view z_scheme_factors[(z_level_idx+1):end]` (no per-stage `Int[...]` vector alloc).

### 3b — Heavy per-(h,z) mask / temp allocations

- **Status:** **Done**  
  - Fused drop mask written into a **reused `BitArray`** (`combined_mask_buf` + `view` per stage).  
  - **`tke` / `var_*` / `cov_*`** use **preallocated `Array{FT,3}` slabs** per horizontal level + **`view`** per z-stage.  
  - **`flatten_and_filter!`:** single column-major pass, preallocated column vectors — **no `findall`**, no 20× re-scan of the grid.

### 3c — `q_con = v_ql .+ v_qi` allocation

- **Status:** **Done** — `CoarseGraining.identify_empty_z_levels_from_ql_qi` (max of `ql+qi` per plane, no broadcast temp).

### 3d — Multiple native-`z` spans ⇒ multiple full `process_abstract_chunk` trees

- **Status:** **Documented / deferred** — algorithmic tradeoff (fewer spans vs accuracy). No code change in this batch.

### 3e — `reduce(vcat, Vector{DataFrame})` for case output

- **Status:** **Done**  
  - `process_abstract_chunk_impl`: one accumulator + `append!` instead of `push!` + `vcat`.  
  - `GoogleLES.build_tabular`: `Ref` + `append!` instead of vector of frames + `reduce(vcat, ...)`.

---

## Broader backlog (from earlier analysis)

| Item | Status |
|------|--------|
| Hoist `z_schemes` | Done (3a) |
| Fused drop masks + reused mask slab | Done (3b) |
| `ql+qi` empty-z without temp | Done (3c) |
| Case / chunk `DataFrame` concat | Done (3e) |
| Reuse diag buffers (`tke`, vars, covs) across z-stages | Done |
| `flatten_and_filter!` single-pass gather | Done |
| `MLCD_SKIP_FINITE_ASSERT_PER_CHUNK` | Done |
| Threaded horizontal coarsening kernels | **Deferred** (after serial) |
| Parallel workers (shards per case/timestep) | **Deferred** (after serial) |
| Convolutional / patch output representation | **Research** (schema + fewer rows?) |

---

## How to profile

```bash
julia --project=/path/to/MLCondensateDistributions \
  /path/to/MLCondensateDistributions/experiments/amip_baseline/profile/profile_dataset_builder_breakdown.jl
```

Optional env: `NX`, `NY`, `NZ`, `REPEATS`, `PROFILE=1` (see script header).

### GoogleLES `build_tabular` (real remote case)

Set **`MLCD_GOOGLELES_TIMESTEP_PROFILE=1`** when running `GoogleLES.build_tabular` (e.g. via `experiments/amip_baseline/build_data.jl`). You get:

- Running **average over cloudy timesteps** (same cadence as progress, every 8 processed steps): `qc` / `prep` / `nonqc_zarr` / `tabular` (seconds).
- **Summary** line after all timesteps for that case.
- **`Arrow.write`** wall time after the case file is written.

**`MLCD_GOOGLELES_TIMESTEP_PROFILE_EACH=1`**: print one profile line **per cloudy timestep** (very verbose).

---

## Changelog (this doc + code)

- **2026-04-04:** Implemented 3a, 3b (fused masks), 3c, 3e; added `identify_empty_z_levels_from_ql_qi`; `build_z_level_keep_mask` accepts `AbstractVector{Int}`; tests for ql/qi empty-z parity; `test/Project.toml` now depends on parent package via `[sources]` so `test_build_training_data.jl` runs under `Pkg.test`; profile script `experiments/amip_baseline/profile/profile_dataset_builder_breakdown.jl`.
- **2026-04-04 (later):** Reused diagnostic + mask buffers per horizontal level; rewrote `flatten_and_filter!` (single pass, `AbstractArray{Bool,3}` mask); `MLCD_SKIP_FINITE_ASSERT_PER_CHUNK`; removed unused `_gather_3d*` helpers.
