# Coarsening & reduction refactor — implementation plan

This document specifies a **multi-year / multi-PR** refactor of how we build multiscale training targets from LES fields. It replaces ad hoc `coarsening_mode` branches with **explicit reducer types**, adds **true sliding (convolutional) local moments**, and defines **truncated non-overlapping block tiling** so scales are chosen by **physics (Δh, Δz)** rather than by **divisors of `nx, ny, nz`**.

Cross-reference: `VERTICAL_COARSENING_PLAN.md` (vertical ladder & z-dropping rules) — align z policy here with that doc where overlap exists. **Moments / variance / covariance numerics** (Arrow `var_*`, `cov_*`, `tke`, Chan–Pebay merge): see [`MOMENTS_NUMERICS_PIPELINE.md`](MOMENTS_NUMERICS_PIPELINE.md).

**Related planning artifact:** Cursor keeps a compact YAML-backed tracker for this effort at `~/.cursor/plans/coarsening_reduction_refactor_3b142058.plan.md` (phase checklist, mermaid sketch, code touchpoints). **This repo document remains canonical** for merged behavior and edge cases; the Cursor file is for editor-side tracking. When they diverge, update **this file** after each significant PR.

---

## 0. Progress snapshot (living — update when behavior changes)

**As of 2026-04** (GoogleLES / `DatasetBuilderImpl` tabular path):

| Area | Status | Code / notes |
|------|--------|----------------|
| **User-facing `coarsening_mode`** | **Done** | `:hybrid` (default), `:block`, `:sliding` only. `:binary`, `:convolutional`, legacy aliases → **`ArgumentError`** in `process_abstract_chunk_impl`, or **`MLCD_COARSENING_MODE`** legacy strings → **warn once + `:hybrid`** in `tabular_build_options_from_env`. |
| **Explicit block factors** | **Done** | `spatial_info.block_triples` (optional); legacy key `convolutional_square_horizontal` → prefer **`block_square_horizontal`**. |
| **Horizontal block schedule** | **Done** | Default: **`subsample_closed_range(nh_min, nh_max, sliding_window_budget_h)`** (same budget knob as sliding/hybrid; default **5**). Opt-in full band: `horizontal_budget = nothing` in `ReductionSpecs` or set budget **≥** `nh_max - nh_min + 1`. **Tower:** `_block_truncated_horizontal_cache!` chains largest cached `s | nh`. |
| **Horizontal tower + cache (block path)** | **Done** | `_block_truncated_horizontal_cache!` in `dataset_builder_impl.jl`: per-`nh` means/products from native or **largest divisor** in cache; products chain with **`coarsen_fields_at_level`** on `<xy>` caches (not `coarsen_products_at_level`). Then vertical **`fz`** per triple. Non-square explicit triples still use one-shot `coarsen_fields_3d_block`. |
| **Vertical `fz` in block triples** | **Done** | All **`fz \| nz`** with `fz * dz_ref ≤ max_dz` (not binary-only). |
| **Sliding** | **Done (sparse default)** | Valid-box means/products; **`valid_box_anchor_starts`** when strides not explicit (corner-preserving `K` samples; **`K` capped** to number of valid placements). **`uniform_stride_for_valid_box`** when **`sliding_stride_*`** set. |
| **Hybrid** | **Done** | Block pass + sliding extras from `hybrid_sliding_extra_sizes_default` / optional `hybrid_sliding_extra_sizes`; same `z_factors` as block for sliding loop. |
| **Reducer types** | **Partial** | `AbstractReductionSpec` + structs live in **`utils/reduction_specs.jl`**. Orchestration is still **`if coarsening_mode`** in `dataset_builder_impl.jl`, not `process_chunk(::AbstractReductionSpec)`. |
| **Docs / env** | **Partial** | `docs/googleles_build_tabular.md`, `experiments/amip_baseline/README.md` document modes and `MLCD_*`. `dataset_spec.md` / root README may still need grep for stale mode names. |
| **Moments numerics** | **Done (2026-04)** | `docs/MOMENTS_NUMERICS_PIPELINE.md`: schema vs internal stats; Float64 accumulators in block/valid-box kernels; block tower uses Chan/Pebay merge (`utils/statistical_methods/`, `coarsening_pipeline.jl`, `dataset_builder_impl.jl`). |

**Pivots vs early drafts**

- **No “divisor enumeration from native”** as a user mode: removed; use `:block` + **`block_triples`** if a fixed schedule is required.
- **Sliding default** prioritizes **anchor starts** (first/last valid flush) over **uniform stride** when strides are not explicit — see `array_utils.jl` / plan §5.5 discussion.
- **`build_horizontal_multilevel_views`** remains a **staged** API; production block path now **embeds the same chaining idea** for tabular builds.

---

## 1. Goals

- [ ] **Reducer dispatch**: One clear API (types or traits) so call sites choose `BlockReduction`, `SlidingConvolution`, or `HybridReduction` **without** `if coarsening_mode` soup. *(Types exist; dispatch still symbolic — **partial**.)*
- [x] **Block reductions**: Non-overlapping blocks; **truncated domain** — `⌊nx / nh⌋ × nh` per axis, remainder discarded (see `truncated_block_extent`, block path in `dataset_builder_impl.jl`).
- [x] **Sliding convolution**: Valid-box local means/products implemented (`conv3d_valid_box_*`, `*_at_starts`); **does not** reuse block cache DAG (by design). Further perf (threading, etc.) still open.
- [x] **Hybrid (default)**: Block tower + sliding gap passes; default **`TabularBuildOptions.coarsening_mode = :hybrid`** and env default **`hybrid`**.
- [x] **Deprecate `binary` / `convolutional` as user modes**: **Removed** from API (errors + env remap); default horizontal block `nh` is **subsampled** in **`[nh_min,nh_max]`**, not divisor-of-native enumeration.
- [~] **Tests & perf**: Many unit/integration tests; **large-N perf regression** vs stride-1 baseline (§9 Phase 5b) still open.
- [x] **Terabyte-scale defaults**: Sparse sliding outputs (`sliding_outputs_*` default 2), window budget (`sliding_window_budget_h`), subsampled sliding triples — see Phase 5b.

Non-goals (initially):

- [ ] Learned convolutions / trainable kernels.
- [ ] Distributed-memory parallelism (MPI); focus on single-process Julia.

---

## 2. Current state (shipped tabular builder)

- [x] **`DatasetBuilderImpl.process_abstract_chunk_impl`** branches on **`coarsening_mode`**: `:hybrid` → block truncated + hybrid sliding; `:block` / `:block_truncated` → block only; `:sliding` → sliding triples only. Removed modes **throw** `ArgumentError` with migration text.
- [x] **`TabularBuildOptions`**: `coarsening_mode`, `sliding_outputs_h|v|z`, `sliding_window_budget_h`; **`tabular_build_options_from_env`** reads **`MLCD_COARSENING_MODE`**, **`MLCD_SLIDING_*`**, etc.
- [x] **`spatial_info`**: `coarsening_mode`, `min_dh`, `block_triples` (optional), `block_square_horizontal` (legacy `convolutional_square_horizontal`), `sliding_stride_*`, `sliding_outputs_*`, `sliding_window_budget_h`, optional `hybrid_sliding_extra_sizes`.
- [ ] **Touchpoint audit**: periodic **`rg coarsening_mode|convolutional|binary`** across `experiments/`, `docs/`, `README` — keep migration table (§8) current.

---

## 3. Reducer taxonomy (dispatch targets)

Define **immutable** config types (names negotiable):

| Type | Semantics | Output grid |
|------|-----------|-------------|
| `BlockReductionSpec` | Non-overlapping `nh×mh×nz` blocks on **truncated** subdomain | `⌊nx/nh⌋ × ⌊ny/mh⌋ × ⌊nz/nz_blk⌋` (per application) |
| `SlidingConvolutionSpec` | Valid box, stride `s` — **default is sparse** (few origins per axis, e.g. **2** to cover left/right); optional dense `s = 1` for research | See §5 |
| `HybridReductionSpec` | Holds `BlockReductionSpec` schedule + `SlidingConvolutionSpec` targets for **gap scales** | Union of outputs (tagged by `reduction_kind` metadata) |

**Dispatch rule (target)**: `process_chunk(fields, reducer::AbstractReductionSpec, ...)` — **one** internal pipeline per type; hybrid orchestrates the other two. **Today:** symbolic `coarsening_mode` + shared helpers from `ReductionSpecs` / `CoarseningPipeline`.

Checklist:

- [x] Types live in **`utils/reduction_specs.jl`**; consumed from **`dataset_builder_impl.jl`** (and re-exported patterns via `coarsening_pipeline.jl` where applicable).
- [~] Arrow metadata: **`reduction_kind`**, **`reduction_nh`**, **`reduction_fz`**, **`truncation_x/y/z`** emitted; **`stride_*` resolved** not always duplicated as columns — see `dataset_builder.jl` schema. Optional future: `native_nx` debug columns (§6).

---

## 4. Block reductions — truncated tiling

### 4.1 Horizontal example (truncation geometry — illustrative)

Domain **124×124** native, `dx = 50 m`, target **≥ 1 km** ⇒ **`nh_min = 20`**, **`nh_max = 124`**. **Default** block `nh` list is **five** evenly spaced integers in `[20,124]` (e.g. **20, 46, 72, 98, 124** with default budget **5**). **Full** ladder `20:124` requires a larger `sliding_window_budget_h` (≥ 105) or `horizontal_budget = nothing` from Julia. The **`nh = 21`** subsection below illustrates **truncation geometry** for that width (not default schedule).

For **`nh = 21`**:

- `⌊124 / 21⌋ = 5` blocks along x → **105** columns used, **19** discarded — **fixed low-index origin**: columns `1:105`, discard `106:124`.
- Same along y.

### 4.2 Rules

- [x] **Origin**: Fixed low-index corner **`(1,1,1)`** (truncated block / valid-box conventions in `dataset_builder_impl` / `array_utils`).
- [x] **Remainder**: Dropped; **no** renormalization of global means by “fraction of domain.”
- [x] **Vertical**: `⌊nz / fz⌋` full layers from bottom; remainder dropped (consistent with `conv3d_block_mean` / vertical coarsen).
- [x] **Schedule**: Horizontal **`nh`** default = **subsampled** (`sliding_window_budget_h` points) in **`[ceil(min_h/dx), min(nx,ny)]`**; optional **full** integer band via large budget or `horizontal_budget = nothing`. Vertical **`fz`**: **divisors of `nz`** with **`fz * dz_ref ≤ max_dz`**.
- [x] **Chaining / cache** (horizontal block path): `_process_abstract_chunk_block_truncated` builds per-`nh` means/products from native or from the largest cached divisor `s | nh` (pool by `nh÷s`); products chain with `coarsen_fields_at_level` on cached `<xy>` fields (not `coarsen_products_at_level`, which expects native field keys). Vertical `fz` is applied per triple after horizontal cache lookup.

### 4.3 “Largest block nh ≤ N/2” and the hybrid gap

Non-overlapping blocks of size `nh` can only tile at most **`⌊N/nh⌋`** windows; the **effective** coverage gap the user cares about is: **desired physical scales** that do **not** correspond to an integer `nh` that tiles **and** meets `min_h`. **Sliding** convolution **can** produce outputs at **many** origins (stride 1) for research, but **defaults** use **sparse** origins (§5.5) with window sizes from a **small schedule** (§5.6), filling scales **between** block-only ladders without dense grids.

Hybrid default (conceptual):

- [x] **Phase A — blocks (schedule)**: `truncated_horizontal_sizes` default subsamples **`nh_min:nh_max`** to **`sliding_window_budget_h`** points. Hybrid sliding extras skip sizes already in the block set; filtered **`default_hybrid_sliding_windows`** avoids duplicating a block `nh`.
- [x] **Phase B — sliding (operational)**: Hybrid runs **`hybrid_sliding_extra_sizes_default`** (subsampled window sizes **not** already in the block `nh` set, budget **`sliding_window_budget_h`**) × same **`z_factors`** as block schedule; **always from native** fine fields for sliding passes (no block-cache reuse). **Not yet**: explicit “Δh tolerance” planner (e.g. 5%) — current policy is **subsampled extras**, not meter-gap optimization.

Document **tolerance** (e.g. 5% on Δh) in config — **future**.

### 4.4 Default relationship: blocks vs sliding

- **Non-overlapping windows** along a direction are **mathematically the same family** as **truncated block reduction** (§4). The **default pipeline** should **prefer the block path** for those scales — it is cheaper and matches the “tower” story.
- **Sliding** is for **overlap** (smaller stride than block width), **gaps** (stride larger than block width so windows do not touch), or **large windows** where **at most one** non-overlapping tile fits — in the latter case the **default sparse sliding** collapses to something like **two origins per axis** (e.g. **2×2** placements in the horizontal plane) so the domain is still **sampled** without a stride-1 quadratic output grid.
- Users can still request **denser** sliding (more outputs, smaller stride) when they need it; defaults optimize for **throughput** over sub-pixel shifts of the window.

---

## 5. Sliding convolution (naive → fast)

### 5.1 Quantities

For each output cell `(i,j,k)` and window `Wh×Wh×Wz` (square horizontal default):

- [x] **First-order** fields: local **mean** per valid box (`conv3d_valid_box_mean`, `*_at_starts`).
- [x] **Products** for covariances / TKE: local **mean of `f*g`** on the same window (`conv3d_valid_box_product_mean`, `*_at_starts`).
- [x] Same derived diagnostics as block path (`StatisticalMethods.covariance_from_moments!`, `Dynamics.tke_field_from_velocity_moments!`, etc.) **per output cell**.

### 5.2 Naive implementation

- [x] Correctness: valid-box loops in **`array_utils.jl`**; tests e.g. **`test_array_utils.jl`**, **`test_dataset_builder_impl`** / pipeline tests.
- [~] Standalone “hand-derived micro-grid” reference test for sliding — optional tighten-up.

### 5.3 Performance (iterative checklist)

- [ ] **Thread** over output `i` or `(i,j)` tiles.
- [x] **Contiguous memory**: column-major 3D arrays; preallocated scratch in hot paths where done.
- [x] **Precompute** `inv(window_volume)` where applicable in box kernels.
- [ ] Consider **separable horizontal** box filter = two passes 1×Wh and Wh×1 (exact for mean; verify for **products** — `mean(f*g)` is **not** separable unless approximated; **must not** use separable trick for cross-moment without proof). *Default: full 3D box for products.*
- [ ] Optional: **FFT / imfilter** only where exact equivalence holds (likely **means only**); document if used.

### 5.4 Boundaries

- [x] **Valid** box only (no padding / ghost) — matches block truncation spirit.
- [x] Output sizes: **`valid_box_output_extent`**, **`uniform_stride_for_valid_box`** docstrings; anchor path uses **explicit start indices** (`valid_box_anchor_starts`).

### 5.5 Stride / output-count defaults (sparse coverage)

**Goal:** Few origins — stride-1 output grids are **not** the default.

- [x] **Target count `K` per axis**: `sliding_outputs_h|v|z` (default **2**), env **`MLCD_SLIDING_OUTPUTS_*`**.
- [x] **User override:** **`sliding_stride_h|v|z`** in `spatial_info` → **`uniform_stride_for_valid_box`**-style strided valid box (integer stride; see `array_utils` / `_resolve_sliding_strides` in `dataset_builder_impl`).
- [x] **Default when strides not explicit:** **`valid_box_anchor_starts`**: first start **1**, last start **`n − window + 1`**, interior via integer spacing; **length `min(K, L)`** if only **`L`** valid placements exist (single-tile domain).
- [~] **`uniform_stride_for_valid_box`**: caps outputs ≤ `K` but **does not** force far-edge flush when `(n−window)` not divisible by `K−1` — documented in `array_utils.jl`; **anchor** path is default for “corners + exact `K` when possible.”
- [ ] Record **resolved** strides / anchor lists in Arrow metadata (optional future; truncation fields exist for blocks).

### 5.6 Window-size schedule (avoid enumerating every width)

Enumerating **every** horizontal window size is **not** the default.

- [x] **Budget** **`sliding_window_budget_h`** (default **5**): **`sliding_reduction_triples`**, **`hybrid_sliding_extra_sizes_default`** — subsampled **`nh`** / **`Wh`** between physics min and **`min(nx,ny)`**.
- [x] **Pure `:sliding` mode** uses **`sliding_reduction_triples`** (budgeted), not every integer width.
- [x] **Override:** explicit **`block_triples`** for block path; sliding still uses schedule from **`sliding_reduction_triples`** unless extended later.

---

## 6. Metadata & training contract

Emitted columns (see **`SCHEMA_SYMBOL_ORDER`** in `dataset_builder.jl`):

- [x] **`reduction_kind`** (string): **`"block_truncated"`** (block and hybrid **block** phase), **`"sliding_valid"`** (pure sliding), **`"hybrid_sliding"`** (hybrid **extra** sliding passes). **`flatten_and_filter!`** default placeholder **`"hybrid"`** if callers omit (e.g. profiling helpers).
- [x] **`reduction_nh`**, **`reduction_fz`**, **`truncation_x`**, **`truncation_y`**, **`truncation_z`**
- [x] **`resolution_h`**, **`resolution_z`**, **`domain_h`**
- [ ] **`native_nx`, `native_ny`, `native_nz`**, explicit **`stride_*`** columns — **not** in schema yet (debug / future).

Ensure **dataset readers** and **train scripts** tolerate **`reduction_kind`** string values above.

---

## 7. Math — merging for towers (block path only)

- [x] **Means / mean products**: horizontal tower uses **sequential box pooling** from native or parent scale; **equivalent** to one-shot `nh×nh` mean when dimensions divide (validated by construction; spot tests in pipeline tests).
- [x] **Non-binary horizontal factors**: **integer ratios** `r = nh / best_src` from cached scale; vertical **`fz`** applied after horizontal cache.
- [ ] **Formal proof / Chan–Welford** in docstrings for merge algebra — still **nice-to-have**.
- [ ] **k-way merge** in one kernel vs repeated 2-way — **not** required for current schedule.

Sliding path: **no merge** across origins; block and sliding caches **not** shared.

---

## 8. Migration from `coarsening_mode`

| Previous | **Shipped (2026)** |
|----------|---------------------|
| `:binary`, `:convolutional`, legacy symbols | **`ArgumentError`** in `process_abstract_chunk_impl` with text pointing to **`:hybrid`**, **`:block`**, **`:sliding`** and **`block_triples`**. |
| `MLCD_COARSENING_MODE=binary|convolutional|…` | **`@warn` once** ( **`maxlog=1`** ) and treat as **`:hybrid`** in `tabular_build_options_from_env` — user should fix env. |
| `convolutional_triples`, `convolutional_square_horizontal` | Use **`block_triples`**, **`block_square_horizontal`** (legacy horizontal key still read as fallback). |
| `MLCD_REDUCTION_MODE` rename | **Not done** — still **`MLCD_COARSENING_MODE`**; rename optional. |
| `TabularBuildOptions` | **`coarsening_mode`**, **`sliding_outputs_*`**, **`sliding_window_budget_h`**; **`tabular_build_options_from_env(; kw...)`** merge pattern. |

Checklist:

- [~] Grep **`experiments/`**, **`docs/`**, **`README`**, **`dataset_spec.md`** for stale **`binary` / `convolutional`** instructions — **ongoing** hygiene.
- [x] Tests updated to **`:hybrid`** / **`:block`** where applicable.
- [x] **Deprecation**: legacy env warns; in-process **`:binary` / `:convolutional`** **hard error** (no silent alias in `spatial_info`).

---

## 9. Phased delivery (PR-sized)

### Phase 0 — Spec & scaffolding

- [x] This doc + team norms for truncation / sparse defaults (evolved in code).
- [x] **`AbstractReductionSpec` + structs** in **`reduction_specs.jl`**.
- [~] **Single `process_chunk(::AbstractReductionSpec)`** entrypoint — **not** done; symbolic routing remains.

### Phase 1 — Block truncated tiling (means only POC)

- [x] Superseded by full block path (all fields + diagnostics).

### Phase 2 — Full diagnostics on block truncated path

- [x] Products, masks, **`flatten_and_filter!`**, Arrow schema — **shipped**.
- [x] **Horizontal tower cache** for means + **product** fields (`_block_truncated_horizontal_cache!`).

### Phase 3 — Sliding naive + tests

- [x] Valid-box kernels + tests (`test_array_utils`, pipeline / dataset tests).
- [x] **`valid_box_anchor_starts`** + **`conv3d_valid_box_*_at_starts`** as default sparse placement.

### Phase 4 — Sliding fast

- [ ] Threading / separable / FFT — **open** (§5.3).

### Phase 5 — Hybrid default

- [x] **Hybrid** orchestration in **`_process_abstract_chunk_hybrid`**.
- [x] Default **`coarsening_mode = :hybrid`** and env default **`hybrid`**.
- [~] **Planner with explicit Δh tolerance** — **not** implemented; subsampled extras only.

### Phase 5b — Sparse sliding + window budgets (operational defaults)

- [x] **`uniform_stride_for_valid_box`**, **`valid_box_anchor_starts`**, **`_resolve_sliding_strides`**, **`MLCD_SLIDING_*`**, explicit **`sliding_stride_*`**.
- [x] **`sliding_reduction_triples`**, **`hybrid_sliding_extra_sizes_default`**, **`sliding_window_budget_h`**.
- [ ] Regression / perf test: large `nx`, `ny` **linear** default sliding cost vs stride-1 baseline.

### Phase 6 — Deprecate & delete legacy

- [x] **Remove** user-facing **`:binary` / `:convolutional`** paths; **env** legacy → warn + **hybrid**.
- [~] Repo-wide doc grep for old names — **ongoing**.

---

## 10. Testing checklist (always-on)

- [~] **Truncation**: covered indirectly; **exhaustive** remainder matrix (0, 1, `nh-1`, …) **optional** expansion.
- [x] **Schedule**: **`truncated_horizontal_sizes`** / **`block_reduction_triples`** tests in **`test_reduction_specs.jl`**.
- [x] **Sliding / anchors**: **`test_array_utils.jl`** (`valid_box_anchor_starts`, `uniform_stride_for_valid_box`, conv vs brute force).
- [x] **Sparse sliding**: **`K` cap** when `L=1` valid placement (`valid_box_anchor_starts`); hybrid / dataset tests pass.
- [x] **Window budget**: **`sliding_reduction_triples`** length ≤ budget test.
- [x] **Regression**: **`test_dataset_builder.jl`**, **`test_dataset_builder_impl.jl`**, **`test_data_hygiene.jl`**, etc.
- [ ] **Thread safety**: when threading is added to hot loops — **N/A** until then.

---

## 11. Open questions / follow-ups

- [x] **Vertical sliding**: **full 3D** valid box for means/products in **`dataset_builder_impl`** sliding/hybrid paths.
- [ ] **Hybrid planner**: replace subsampled extras with **explicit Δh / Δz gap** rule (meters or relative)?
- [ ] **cfSites** path: parity review vs GoogleLES **`spatial_info`** defaults.
- [ ] **Arrow duplication**: hybrid can emit **similar** `resolution_h` from block vs sliding — keep **`reduction_kind`** distinct or dedupe in planner?

**Defaults (Phase 5b — in code, `dataset_builder_impl` + `reduction_specs` + `array_utils`):**

- **Block-first** hybrid; **sparse sliding** for extra window sizes; default **`K = 2`** per axis via **`sliding_outputs_*`** when strides not explicit.
- **Anchor path**: corner-preserving **`valid_box_anchor_starts`**; **`uniform_stride_for_valid_box`** when **`sliding_stride_*`** set (different edge semantics — see §5.5).
- **Window sizes:** **`sliding_window_budget_h`** subsamples block `nh`, sliding windows, and hybrid-extra candidates within **`[nh_min,nh_max]`** (full band if budget ≥ span length).

---

## 12. Sign-off

| Item | Owner | Date |
|------|-------|------|
| Truncation origin (corner) | | |
| Hybrid default tolerances | | |
| Env var naming | | |
| Deprecation timeline | | |

---

*Last updated: 2026-04 — default horizontal block `nh` subsampled (`sliding_window_budget_h`); optional full `nh_min:nh_max`; divisor-chain block cache; hybrid sliding extras dedupe; legacy mode removal + env behavior.*
