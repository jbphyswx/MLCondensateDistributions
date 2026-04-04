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
| `SlidingConvolutionSpec` | Box window, stride `s` (default `s = 1` for “true” sliding), valid / same padding policy TBD | See §5 |
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

Non-overlapping blocks of size `nh` can only tile at most **`⌊N/nh⌋`** windows; the **effective** coverage gap the user cares about is: **desired physical scales** that do **not** correspond to an integer `nh` that tiles **and** meets `min_h`. **Sliding** convolution can produce outputs at **every** origin (stride 1) with window size tied to physical Δh, filling scales **between** block-only ladders.

Hybrid default (conceptual):

- [ ] **Phase A — blocks**: Emit all truncated block scales from a **small factor set** (e.g. primes ladder 2,3,5 or user schedule) with **caching**.
- [ ] **Phase B — sliding**: For each **target Δh** (and Δz) **not** hit by Phase A within tolerance, run sliding local moments on **native** (or on a **fine-enough** cached grid — only if bit-exact equivalence proven).

Document **tolerance** (e.g. 5% on Δh) in config.

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
- [ ] Document output sizes: `(nx - Wh + 1) × …` for stride 1 valid.

---

## 6. Metadata & training contract

Every emitted row / column group must record:

- [ ] `reduction_kind`: `:block_truncated` | `:sliding_valid` | `:hybrid_block` | `:hybrid_sliding`
- [ ] `nh`, `mh`, `nz_blk` **or** `window_h`, `window_z`, `stride`
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

- [ ] Land this doc; team sign-off on **truncation corner** and **stride** defaults.
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

### Phase 6 — Deprecate & delete legacy

- [ ] Redirect `:binary` / old `:convolutional` to new specs; remove divisor enumerator.
- [ ] Clean docs; close tickets.

---

## 10. Testing checklist (always-on)

- [ ] **Truncation**: remainder sizes 0, 1, `nh-1`, `nh`, `N-1`.
- [ ] **Prime / composite** `nx` — block schedule still defined by physics, not divisibility.
- [ ] **Sliding**: stride 1 valid; compare means/products to reference.
- [ ] **Regression**: snapshot row counts for a fixed small case (or tolerance on statistics).
- [ ] **Thread safety** (if threaded): same results as serial.

---

## 11. Open questions (resolve before Phase 5)

- [ ] **Vertical sliding**: independent 1D along z or full 3D window? (Recommend full 3D for consistency with moments.)
- [ ] **Hybrid planner**: exact rule for “gap” — absolute meters vs relative to `dx*nh`?
- [ ] **cfSites** path: same reducer specs or separate defaults?
- [ ] **Arrow duplication**: hybrid may emit **two** rows for “similar” Δh from block vs sliding — deduplicate in planner or keep both with distinct `reduction_kind`?

---

## 12. Sign-off

| Item | Owner | Date |
|------|-------|------|
| Truncation origin (corner) | | |
| Hybrid default tolerances | | |
| Env var naming | | |
| Deprecation timeline | | |

---

*Last updated: implementation plan draft for reducer refactor (block / sliding / hybrid).*
