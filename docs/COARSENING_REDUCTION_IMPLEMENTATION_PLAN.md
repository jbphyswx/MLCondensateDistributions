# Coarsening & reduction refactor — implementation plan

This document specifies a **multi-year / multi-PR** refactor of how we build multiscale training targets from LES fields. It replaces ad hoc `coarsening_mode` branches with **explicit reducer types**, adds **true sliding (convolutional) local moments**, and defines **truncated non-overlapping block tiling** so scales are chosen by **physics (Δh, Δz)** rather than by **divisors of `nx, ny, nz`**.

Cross-reference: `VERTICAL_COARSENING_PLAN.md` (vertical ladder & z-dropping rules) — align z policy here with that doc where overlap exists.

---

## 1. Goals

- [ ] **Reducer dispatch**: One clear API (types or traits) so call sites choose `BlockReduction`, `SlidingConvolution`, or `HybridReduction` (name TBD) without `if coarsening_mode` soup.
- [ ] **Block reductions**: Non-overlapping blocks; **truncated domain** — use `⌊nx / nh⌋ × nh` cells per horizontal dimension and **discard the remainder** (no fractional blocks, no periodic wrap, no per-block weights for margin).
- [ ] **Sliding convolution**: **Naive** first pass (correctness), then **performance** passes (threading, cache-friendly loops, optional separable / library kernels) — **no** pretending sliding outputs can reuse the block-reduction cache DAG.
- [ ] **Hybrid (proposed default)**: Run **fast block-reduction towers** where they tile cleanly, and use **sliding windows** to **fill scale gaps** between what blocks can represent (e.g. between `nh` and `⌊nx/2⌋` coverage — exact gap definition in §4.3).
- [ ] **Deprecate `binary` as a user-facing mode**: Internally becomes a **schedule** over **block factors** (powers of two, etc.), not a separate mathematical world.
- [ ] **Tests & perf**: Correctness vs reference (small grids), regression on row counts / statistics, benchmarks on realistic `nx, ny, nz`.
- [ ] **Terabyte-scale defaults**: Processing must stay tractable on huge corpora — **prefer block/non-overlapping** where it applies; for sliding/conv, **default to the fewest useful origins** (large strides, bounded output count per axis) and **a small, linearly spaced set of window sizes** — not dense stride-1 and not every integer window from `N/2` to `N−1`.

Non-goals (initially):

- [ ] Learned convolutions / trainable kernels.
- [ ] Distributed-memory parallelism (MPI); focus on single-process Julia.

---

## 2. Current state (baseline to replace)

- [ ] Documented in code review: `DatasetBuilderImpl.process_abstract_chunk_impl` branches on `coarsening_mode`:
  - **`:binary`**: horizontal seed ladder + **chained** horizontal coarsening; vertical **×2** ladder from `dz` scheme; reuses intermediates.
  - **`:convolutional`**: enumerates divisor triples `(fx, fy, fz)`; each triple **re-pools from native** — flexible shapes but **expensive** and **grid-dependent**.
- [ ] `TabularBuildOptions.coarsening_mode` + `spatial_info` keys (`convolutional_triples`, etc.) — list all touchpoints in a migration table (§8).

---

## 3. Reducer taxonomy (dispatch targets)

Define **immutable** config types (names negotiable):

| Type | Semantics | Output grid |
|------|-----------|-------------|
| `BlockReductionSpec` | Non-overlapping `nh×mh×nz` blocks on **truncated** subdomain | `⌊nx/nh⌋ × ⌊ny/mh⌋ × ⌊nz/nz_blk⌋` (per application) |
| `SlidingConvolutionSpec` | Valid box, stride `s` — **default is sparse** (few origins per axis, e.g. **2** to cover left/right); optional dense `s = 1` for research | See §5 |
| `HybridReductionSpec` | Holds `BlockReductionSpec` schedule + `SlidingConvolutionSpec` targets for **gap scales** | Union of outputs (tagged by `reduction_kind` metadata) |

**Dispatch rule**: `process_chunk(fields, reducer::AbstractReductionSpec, ...)` or equivalent — **one** internal pipeline per type; hybrid orchestrates the other two.

Checklist:

- [ ] Types live in a small module (e.g. `utils/reduction_specs.jl`) included from `coarsening_pipeline.jl` / `dataset_builder_impl.jl`.
- [ ] JSON/Arrow metadata includes `reduction_kind`, `nh`, `mh`, `nz_blk`, `stride`, `truncation` (see §6).

---

## 4. Block reductions — truncated tiling

### 4.1 Horizontal example (user spec)

Domain **124×124** native, `dx = 50 m`, target **≥ 1 km** ⇒ need `nh × 50 m ≥ 1000` ⇒ `nh ≥ 20`. Pick **`nh = 21`**:

- `⌊124 / 21⌋ = 5` blocks along x → **105** columns used, **19** columns **discarded** (e.g. right edge; document **which edge** — recommend **fixed low-index origin**: use columns `1:105`, discard `106:124`).
- Same along y.

### 4.2 Rules

- [ ] **Origin**: Fixed corner (document: e.g. **start at `(1,1,1)`** in Julia 1-based indices).
- [ ] **Remainder**: Dropped; **no** renormalization of global means by “fraction of domain.”
- [ ] **Vertical**: Same idea: `⌊nz / nz_blk⌋` full layers; drop top remainder unless we explicitly choose bottom vs top (pick one and test).
- [ ] **Schedule**: Build a list of `(nh, mh, nz_blk)` from **physical targets** (`min_h`, `max_dz`, etc.), not from **divisors** of `nx`.
- [ ] **Chaining / cache**: Prefer **tower** along factors (2×2, 3×3, …) so intermediate scales are reused — **means and moment fields** must use merge rules that match emitted diagnostics (see §7).

### 4.3 “Largest block nh ≤ N/2” and the hybrid gap

Non-overlapping blocks of size `nh` can only tile at most **`⌊N/nh⌋`** windows; the **effective** coverage gap the user cares about is: **desired physical scales** that do **not** correspond to an integer `nh` that tiles **and** meets `min_h`. **Sliding** convolution **can** produce outputs at **many** origins (stride 1) for research, but **defaults** use **sparse** origins (§5.5) with window sizes from a **small schedule** (§5.6), filling scales **between** block-only ladders without dense grids.

Hybrid default (conceptual):

- [ ] **Phase A — blocks**: Emit all truncated block scales from a **small factor set** (e.g. primes ladder 2,3,5 or user schedule) with **caching**.
- [ ] **Phase B — sliding**: For each **target Δh** (and Δz) **not** hit by Phase A within tolerance, run sliding local moments on **native** (or on a **fine-enough** cached grid — only if bit-exact equivalence proven).

Document **tolerance** (e.g. 5% on Δh) in config.

### 4.4 Default relationship: blocks vs sliding

- **Non-overlapping windows** along a direction are **mathematically the same family** as **truncated block reduction** (§4). The **default pipeline** should **prefer the block path** for those scales — it is cheaper and matches the “tower” story.
- **Sliding** is for **overlap** (smaller stride than block width), **gaps** (stride larger than block width so windows do not touch), or **large windows** where **at most one** non-overlapping tile fits — in the latter case the **default sparse sliding** collapses to something like **two origins per axis** (e.g. **2×2** placements in the horizontal plane) so the domain is still **sampled** without a stride-1 quadratic output grid.
- Users can still request **denser** sliding (more outputs, smaller stride) when they need it; defaults optimize for **throughput** over sub-pixel shifts of the window.

---

## 5. Sliding convolution (naive → fast)

### 5.1 Quantities

For each output cell `(i,j,k)` and window `Wh×Wh×Wz` (square horizontal default):

- [ ] For every **first-order** field `f`: store **local mean** `mean(f)`.
- [ ] For every **product** needed for covariances / TKE: store **local mean** `mean(f*g)` on the **same window**.
- [ ] Emit same derived diagnostics as block path (`_covariance_from_moments!`, `_tke_from_moments!`, etc.) **per output cell**.

### 5.2 Naive implementation

- [ ] Six nested loops or equivalent: `for i,j,k` over output, `for di,dj,dk` over window — **correctness reference**.
- [ ] Unit test: compare to explicit small-grid hand calculation.

### 5.3 Performance (iterative checklist)

- [ ] **Thread** over output `i` or `(i,j)` tiles.
- [ ] **Contiguous memory**: ensure arrays are column-major friendly.
- [ ] **Precompute** `inv(window_volume)` once per scale.
- [ ] Consider **separable horizontal** box filter = two passes 1×Wh and Wh×1 (exact for mean; verify for **products** — `mean(f*g)` is **not** separable unless approximated; **must not** use separable trick for cross-moment without proof). *Default: full 3D box for products.*
- [ ] Optional: **FFT / imfilter** only where exact equivalence holds (likely **means only**); document if used.

### 5.4 Boundaries

- [ ] **Valid** convolution only (shrink output) for v1 — matches “no ghost / no periodic” spirit of block truncation.
- [ ] Document output sizes: for stride `s`, horizontal extent is `⌊(nx − Wh) / s⌋ + 1` (and analogously `y`, `z`) when `nx ≥ Wh`; zero outputs if `nx < Wh`.

### 5.5 Stride / output-count defaults (sparse coverage)

**Goal:** Few origins, large effective step — **tiny shifts of the window add little information** relative to the volume of LES data; stride-1 output grids are **quadratic** in domain extent and are **not** the default.

- [ ] Expose a **target output count per axis** `K` (default **`2`**: first valid window at the **low-index** origin, last aligned toward the **far edge** — e.g. `Wh = 70`, `nx = 124` ⇒ stride `54` ⇒ **2** starts along `x`).
- [ ] **User override:** explicit integer **strides** `stride_h`, `stride_v`, `stride_z` (or env / `spatial_info`) for **overlap** or **gaps**.
- [ ] **Divisibility / `K > 2`:** Prefer **approximately even coverage** of the domain along each axis. Use a **simple rule**: derive a **floating** ideal spacing `(nx − Wh) / (K − 1)`, map to **integer stride** by **rounding** (document whether `round` / `floor` / `ceil`), then **adjust** so the realized output count **never exceeds `K`** (avoid extra work — **more than `K` outputs is unacceptable** for the default budget). Small **asymmetry** at the right/top edge is acceptable if it preserves the output cap and near-uniform spacing.
- [ ] Record resolved strides and target `K` in metadata (§6).

### 5.6 Window-size schedule (avoid enumerating every width)

Enumerating **every** horizontal window size from **`⌊N/2⌋` to `N−1`** (e.g. **65…123** on **124×124**) is **another quadratic-style blow-up**. **Block reductions** already fill many scales; sliding/convolution should **not** repeat that enumeration by default.

- [ ] Add a configurable **budget** of window sizes (default on the order of **`5`**, linearly or **evenly spaced** in index space between a **minimum** from physics (`min_h`, `dx`, `min_dz` / `dz_ref`) and a **maximum** (e.g. `min(nx, ny)` or policy “below full domain” so blocks own the largest non-overlapping scales).
- [ ] Apply the same idea to **pure convolution / sliding modes** that iterate over window sizes: **default = subsampled schedule**, not all integers.
- [ ] **Override:** user supplies an explicit list of `(Wh, Wz)` or triples when they need full coverage.

---

## 6. Metadata & training contract

Every emitted row / column group must record:

- [ ] `reduction_kind`: `:block_truncated` | `:sliding_valid` | `:hybrid_block` | `:hybrid_sliding`
- [ ] `nh`, `mh`, `nz_blk` **or** `window_h`, `window_z`, `stride` — plus optional `sliding_outputs_h` / resolved stride after subsampling policy
- [ ] `resolution_h`, `resolution_z` (physical, as today)
- [ ] `truncation_x`, `truncation_y`, `truncation_z` (counts dropped)
- [ ] `native_nx`, `native_ny`, `native_nz` (for debugging)

Ensure **dataset readers** and **train scripts** tolerate new fields / enum values.

---

## 7. Math — merging for towers (block path only)

- [ ] **Means**: composable under disjoint refinement (same block sizes).
- [ ] **Variances / covariances / TKE**: require **sufficient statistics** per node (`n`, means, mean products, etc.) if merging coarse blocks into coarser without returning to native — mirror existing moment pipeline.
- [ ] **Proof sketch / citation** in docstrings (Chan/Welford / parallel merge).
- [ ] **Non-binary factors**: merging 3 or 4 children = repeated pairwise merge or single k-way formula — pick one implementation style and test.

Sliding path: **no merge** across windows at different origins (unless future “integral images” / summed-area tables for box filters on **means** only — optional future work).

---

## 8. Migration from `coarsening_mode`

| Current | Target |
|---------|--------|
| `:binary` | Deprecated alias → `BlockReductionSpec` with **legacy schedule** (seed ladder + vertical ×2) until tests pass on new tower |
| `:convolutional` | Remove “all divisors from native” behavior; either map to `BlockReductionSpec` + truncated tiling **or** `SlidingConvolutionSpec` per user choice |
| `MLCD_COARSENING_MODE` | Replace with e.g. `MLCD_REDUCTION_MODE=block|sliding|hybrid` + JSON or separate env for schedules |
| `TabularBuildOptions` | Add `reduction_spec` or flattened fields; keep `tabular_build_options_from_env` merge pattern |

Checklist:

- [ ] Grep entire repo for `coarsening_mode`, `convolutional`, `binary` in `spatial_info`.
- [ ] Update `dataset_spec.md` / README snippets if they mention old modes.
- [ ] Deprecation warnings for one release cycle.

---

## 9. Phased delivery (PR-sized)

### Phase 0 — Spec & scaffolding

- [ ] Land this doc; team sign-off on **truncation corner**, **sparse stride / output-count defaults** (§5.5), and **window-size budget** (§5.6).
- [ ] Introduce `AbstractReductionSpec` + structs **without** wiring full pipeline.
- [ ] Add **no-op** or stub dispatch that errors with clear message.

### Phase 1 — Block truncated tiling (means only POC)

- [ ] Implement truncated crop + non-overlapping mean for **one** field.
- [ ] Tests: 124×124, `nh=21` → inner size 5×5; remainder 19 discarded.
- [ ] Extend to full `fields` NamedTuple and **one** scale in `process_abstract_chunk`.

### Phase 2 — Full diagnostics on block truncated path

- [ ] Wire products, cloud mask, flatten to Arrow — match existing column schema.
- [ ] Performance: optional tower cache for means + moments.

### Phase 3 — Sliding naive + tests

- [ ] Implement §5.1–5.2; parity tests on tiny grids vs brute force.
- [ ] Document output shape vs block path.

### Phase 4 — Sliding fast

- [ ] §5.3 items; benchmark vs Phase 3 on realistic size.

### Phase 5 — Hybrid default

- [ ] Implement `HybridReductionSpec` planner: choose block scales + sliding scales from `min_h`, `max_dz`, tolerances.
- [ ] Make **hybrid** the default in `tabular_build_options_from_env`.
- [ ] Integration test: one LES slice (or mocked data) end-to-end.

### Phase 5b — Sparse sliding + window budgets (operational defaults)

- [x] Implement §5.5–5.6: `uniform_stride_for_valid_box` in [`utils/array_utils.jl`](utils/array_utils.jl); `_resolve_sliding_strides` + `spatial_info` / `TabularBuildOptions` (`sliding_outputs_*`, `sliding_window_budget_h`, env `MLCD_SLIDING_*`); explicit `sliding_stride_*` overrides auto stride.
- [x] Subsample sliding **window sizes**: [`sliding_reduction_triples`](utils/reduction_specs.jl), [`hybrid_sliding_extra_sizes_default`](utils/reduction_specs.jl) vs block ladder; `sliding_window_budget_h` (default 5).
- [ ] Regression / perf test: large `nx`, `ny` shows **linear** scaling in domain size for default sliding cost vs stride-1 baseline.

### Phase 6 — Deprecate & delete legacy

- [ ] Redirect `:binary` / old `:convolutional` to new specs; remove divisor enumerator.
- [ ] Clean docs; close tickets.

---

## 10. Testing checklist (always-on)

- [ ] **Truncation**: remainder sizes 0, 1, `nh-1`, `nh`, `N-1`.
- [ ] **Prime / composite** `nx` — block schedule still defined by physics, not divisibility.
- [ ] **Sliding**: stride 1 valid (reference / opt-in dense mode); compare means/products to reference.
- [ ] **Sparse sliding**: default `K = 2` matches hand-computed corner windows; `K > 2` respects **≤ K** outputs and approximate spacing.
- [ ] **Window schedule**: default budget yields **≤ configured** distinct `Wh` (and `Wz`) values.
- [ ] **Regression**: snapshot row counts for a fixed small case (or tolerance on statistics).
- [ ] **Thread safety** (if threaded): same results as serial.

---

## 11. Open questions (resolve before Phase 5)

- [ ] **Vertical sliding**: independent 1D along z or full 3D window? (Recommend full 3D for consistency with moments.)
- [ ] **Hybrid planner**: exact rule for “gap” — absolute meters vs relative to `dx*nh`?
- [ ] **cfSites** path: same reducer specs or separate defaults?
- [ ] **Arrow duplication**: hybrid may emit **two** rows for “similar” Δh from block vs sliding — deduplicate in planner or keep both with distinct `reduction_kind`?

**Defaults (Phase 5b — implemented in code, see `dataset_builder_impl` + `reduction_specs`):**

- Prefer **block** for non-overlapping scales; **sparse sliding** when overlap/gaps or **large windows** demand it; **default `K = 2`** origins per axis unless configured otherwise.
- **`K > 2`:** even spacing with **integer stride rounding**, **never more than `K` outputs**; slight edge asymmetry OK.
- **Window sizes:** default **~5** evenly spaced candidates — not every integer; **block path** fills most scales operationally.

---

## 12. Sign-off

| Item | Owner | Date |
|------|-------|------|
| Truncation origin (corner) | | |
| Hybrid default tolerances | | |
| Env var naming | | |
| Deprecation timeline | | |

---

*Last updated: added terabyte-scale defaults — sparse strides (§5.5), subsampled window schedules (§5.6), block-first vs sliding (§4.4), Phase 5b.*
