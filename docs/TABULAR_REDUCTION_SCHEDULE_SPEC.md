# Tabular multiscale reduction schedule — specification

This document defines **target** behavior for building GoogleLES-style **tabular Arrow** training rows: which horizontal and vertical scales are produced, and how **non-overlapping block** coarsening and **valid-box (sliding)** coarsening interact. It is written for someone with **no prior context** on this repository.

**Scope.** The production entry point is `DatasetBuilderImpl.process_abstract_chunk` (and helpers in `utils/reduction_specs.jl`, `utils/dataset_builder_impl.jl`). **Treat this document as approved-by-stakeholders before changing schedules in code** (§9). Until then, shipped behavior may **not** match every detail here; this file is the **normative target** to implement and test against.

**Related code (reference).** Block extent helper: `truncated_block_extent` / `truncated_block_extent`-style logic (only `⌊n/h⌋·h` cells from the **low-index** side are used; the rest is discarded). Subsample helper: `ReductionSpecs.subsample_closed_range(lo, hi, n)`.

---

## 1. Overview

The pipeline builds **one table** of per-cell (per coarse patch) statistics. Each row corresponds to a **cloudy** coarse location at a chosen **horizontal scale** and **vertical scale** (coarsening factor **`fz`**: native layers per coarse vertical cell).

### 1.1 Horizontal scales (two bands)

1. **Lower band — block reductions.** Non-overlapping **`nh × nh`** averages on a **fixed-origin, contiguous** subdomain (§2.2). **Seeds** are **`B`** subsampled values in **`[nh_min, nh_half]`**. From each seed, **DFS** applies **prime** macro-pools **`p × p`** on the **already-coarsened** grid (§3).

2. **Upper band — valid-box (sliding).** Windowed means with **valid** placement. **`B`** subsampled window widths **`wh`** in **`[max(nh_min, nh_half), N]`**, minus scales already produced as blocks (§4).

### 1.2 Vertical scales — same *shape*, different physics (two bands)

Vertical is **deliberately parallel** to horizontal: **lower band = block + tower**, **upper band = sliding extras in hybrid**, same **`B`** budget idea, **low-index truncation**, **prime** atomic steps — but **1D along `z`** instead of **2D in `(x,y)`**, and driven by a **maximum** coarsened thickness (**`max_dz`**) rather than a **minimum** horizontal footprint (**`min_h_resolution`**).

| Aspect | Horizontal | Vertical |
|--------|------------|----------|
| Finest scale | **`nh`** down to physics floor **`nh_min`** | **`fz = 1`** (one native layer per coarse cell; no extra “min vertical cells” knob) |
| Coarse cap | Domain size **`N`** | **`fz_max = min(nz, fz_cap)`** where **`fz_cap`** is the largest **`fz`** with **`fz · dz_ref ≤ max_dz`** |
| Lower-band seeds | **`[nh_min, nh_half]`**, **`B`** subsamples | **`[1, fz_half]`**, **`B`** subsamples (`§5.1`) |
| Tower moves | Prime **`p`**, **`p × p`** pool | Prime **`p`**, **`p`**-layer mean along **`z`** only |
| Upper band (hybrid) | **`wh ∈ [max(nh_min, nh_half), N]`** subsampled, minus **`block_nh_set`** | **`wz ∈ [fz_half+1, fz_max]`** subsampled, minus **`block_fz_set`** (`§5.2`) |

**Modes:** `:block` (horizontal + vertical **block** bands only), `:sliding` (valid-box only; vertical schedule in §5.3), `:hybrid` (block pass then sliding pass, with **deduplication** so the same horizontal **`wh`** or vertical **`wz`** is not redundantly emitted when it already appeared as a block scale).

---

## 2. Definitions

### 2.1 Symbols

| Symbol | Definition |
|--------|------------|
| `nx`, `ny`, `nz` | Native field dimensions for this **chunk** (one call to the builder). |
| `N` | `min(nx, ny)` — maximum square block width that fits in the horizontal plane. |
| `dx_native` | Native horizontal grid spacing (m), e.g. `(x[end] - x[1]) / (length(x) - 1)` from coordinates. |
| `dz_native_profile` | Length-`nz` vector of native layer thicknesses (m) for this chunk. |
| `dz_ref` | `mean(dz_native_profile)`; used only in the inequality `fz · dz_ref ≤ max_dz`. |
| `min_h_resolution` | Minimum coarse horizontal scale in **meters** (default `1000` for GoogleLES tabular builds). |
| `nh_min` | `max(1, ⌈min_h_resolution / dx_native⌉)` — smallest block width in **cells** that meets the physics minimum. |
| `nh_half` | `⌊N / 2⌋`. |
| `max_dz` | Maximum allowed **mean** vertical layer thickness after coarsening (default `400` m in the tabular path). |
| `fz_cap` | Largest integer **`fz ≥ 1`** such that **`fz · dz_ref ≤ max_dz`** (no dependence on **`nz`**). Example: **`dz_ref = 10` m**, **`max_dz = 400` m** → **`fz_cap = 40`**. |
| `fz_max` | **`min(nz, fz_cap)`** — per chunk, you cannot average more native layers than exist. All vertical schedules use **`fz ∈ {1,…,fz_max}`** (or subsets thereof). |
| `fz_half` | **`⌊fz_max / 2⌋`**. Lower-band tower seeds live in **`1 : fz_half`**; hybrid sliding vertical candidates start at **`fz_half + 1`** (§5). If **`fz_half < 1`**, use **`[1, fz_max]`** for seeds (§5.1). |
| `B` | **Budget** (default `5`): subsample count for horizontal seeds, horizontal sliding widths, vertical tower seeds, and vertical sliding heights (same knob as `sliding_window_budget_h` / tabular options unless a future split is introduced). |

### 2.2 Horizontal block tiling (`x`, `y`)

This is **not** the same as “push convolutions to the corners” in the sliding path.

For **horizontal block** coarsening at native or at any coarse level:

- **Origin:** always the **low-index** corner: horizontal indices start at **`1`** along `x` and `y`.
- **Contiguous blocks:** the domain used is **`1 : nu`** along each horizontal axis, where **`nu = ⌊size / h⌋ · h`** for block width **`h`**.
- **Remainder:** cells **outside** `1:nu` are **discarded** for that scale (not wrapped or centered).

### 2.3 Vertical block tiling (`z`)

Same **spirit** as §2.2, but **one dimension**:

- **Origin:** **`k = 1`** (bottom of the stored column).
- **Contiguous blocks:** use **`nz_used = ⌊nz / fz⌋ · fz`** native layers **`1 : nz_used`**. Each coarse vertical index aggregates **`fz`** consecutive native levels.
- **Remainder:** native levels **`nz_used+1 : nz`** are **not** included in that block mean.

So vertical block coarsening **does not** require **`fz | nz`**. Divisibility was an older shortcut; the **normative** rule is **truncate the high-`k` tail**, analogous to horizontal truncation.

**Sliding** along **`z`** uses valid windows of height **`wz`** inside **`1:nz`** (§4.3); that is separate from this block rule.

### 2.4 Subsampled integer lists (reproducibility)

To pick **`B`** distinct integers in **`[lo, hi]`** approximately evenly spaced, use the same rule as `subsample_closed_range(lo, hi, B)` in `utils/reduction_specs.jl`:

- If `hi < lo`, result is empty.
- If `hi - lo + 1 ≤ B`, result is **`lo:hi`** (every integer).
- Otherwise, for `i = 1, …, B`, take `lo + round((i - 1) * (hi - lo) / (B - 1))`, clamp to `[lo, hi]`, then **sort** and **unique**.

**Example:** `subsample_closed_range(20, 60, 5)` → **`[20, 30, 40, 50, 60]`**.  
**Example:** `subsample_closed_range(60, 120, 5)` → **`[60, 75, 90, 105, 120]`**.

### 2.5 Prime steps for the tower (implementation note)

- **Horizontal:** at each DFS node, valid moves are **prime** **`p`** with a legal **`p × p`** macro-pool (§3.2).
- **Vertical:** at each DFS node, valid moves are **prime** **`p`** with a legal **`p`**-layer pool **along `z` only** (§5.1).

To **enumerate primes** in Julia without hand-maintaining lists, use **[Primes.jl](https://github.com/JuliaMath/Primes.jl)** (e.g. `primes(lo, hi)`, or generate candidates and test `isprime`), or an equivalent sieve. The spec does **not** fix which API to call; it only requires **all primes** in the valid range to be considered, not a hard-coded `{2, 3, 5}` subset.

---

## 3. Lower band — block reductions and tower

### 3.1 Seeds

**Condition:** assume `nh_min ≤ nh_half`. If **`nh_min > nh_half`** (very small horizontal extent), **fallback:** subsample **`[nh_min, N]`** with budget **`B`** only (single band); document in release notes.

**Seed list:**

```text
nh_seeds = subsample_closed_range(nh_min, nh_half, B)
```

These are **only entry points** for DFS; **many** distinct **`nh`** can appear in outputs after the tower.

### 3.2 Tower: prime macro-pools, DFS, global cache

**State:** a global map **`nh →`** (horizontal coarse fields and the parallel **product** tensors used for variances/covariances), only for **`nh`** already computed.

**Step semantics:** At scale **`nh`**, one **legal** prime **`p`** applies a **`p × p`** **non-overlapping block mean** on the **current** coarse grid (with the **same** low-index truncation rule as §2.2). The **native-equivalent** block width becomes **`nh' = nh · p`**.

**Why only prime `p` as atomic moves:** Any composite factor (e.g. **4**, **6**) is a **product of primes**. Reaching **`nh · 4`** is achieved by **`p = 2`** then **`p = 2`** on successive coarse grids, not by introducing a separate **`p = 4`** branch. That keeps the branching set small and matches “try all towers from prime reductions.”

**Validity of prime `p` at node `nh`:** **`p`** is valid if and only if:

1. **`nh · p ≤ N`**, and  
2. After coarsening to **`nh`**, the horizontal shape of the stored field allows **at least one** full **`p × p`** macro-tile from the **low-index** origin under §2.2.

**Algorithm (high level):**

1. For each **`nh₀` in `nh_seeds`** (e.g. increasing order):
   - If **`nh₀`** is absent from the cache, compute it **once** from **native** data with **`nh₀ × nh₀`** blocks; store in cache and **emit** this scale for Arrow (subject to clouds / masks downstream).
2. **DFS(`nh`)** for each seed’s starting **`nh₀`**:
   - For each **prime `p`** in a **fixed iteration order** (spec recommends **ascending `p`** for reproducibility) that is **valid** at **`nh`**:
     - **`nh_child = nh · p`**. If **`nh_child`** is already cached, **do not recompute**; optionally skip duplicate emission.
     - Else compute child by **`p × p`** pooling on the **`nh`** field; cache; emit; **DFS(`nh_child`)**.
   - **Backtrack** when no further valid **`p`** exists (standard DFS).

**Deduplication:** The same **`nh`** may be reachable from **different** seeds or branches; **build and store at most once**.

### 3.3 Block scales beyond `nh_half`

**Default in this spec:** the **tower may produce `nh` up to `N`** with **`reduction_kind = block_truncated`**, because macro-pools can multiply a seed up to the full domain width (e.g. narrative **`20 → … → N`**). The phrase **“lower band”** refers to **where seeds are chosen** (`nh_min` … `nh_half`), **not** that every block output must satisfy **`nh ≤ nh_half`**.

### 3.4 Optional note on composite one-shot pools

For **ideal** full tiling on rectangles whose sizes are **multiples** of all step sizes, nested **`2×2`** then **`2×2`** yields the same **algebraic** mean as one **`4×4`** on the **original** parent grid. With **integer truncation** (§2.2), the **horizontal extent** after the **first** pool may not be divisible by the **second** factor; in those edge cases different **orders** of prime steps could **theoretically** assign **different** sets of native cells to a “same” nominal **`nh'`**. The **default spec** still uses **prime steps only**; a future **optional** “composite-pool” mode is **out of scope** unless benchmarking shows a scientific need.

### 3.5 Worked example (`N = 120`, single seed `nh₀ = 20`)

- Native → **`nh = 20`**.
- **`p = 2`:** **`20 → 40`**.
- From **`40`**, **`p = 2`:** **`40 → 80`**.
- From **`80`**, no valid prime yields **`nh · p ≤ 120`** with a legal tile (e.g. **`2`** gives **160 > N`**). **Backtrack** to **`40`**.
- From **`40`**, **`p = 3`:** **`40 → 120 = N`**.
- Other primes at **`40`** (e.g. **`5`**, **`7`**) fail **`nh·p ≤ N`** or tiling here.

**`nh = 80`** appears as **`2×2` then `2×2`** from **`20`**, not as a single **`p = 4`** move.

---

## 4. Upper band — valid-box (sliding) reductions

### 4.1 Window widths

```text
nh_slide_lo = max(nh_min, nh_half)
nh_slide_hi = N
candidates = subsample_closed_range(nh_slide_lo, nh_slide_hi, B)
```

**Rationale:** sliding windows are **not** used below the physics minimum **`nh_min`**; **`nh_slide_lo`** enforces that when **`nh_half < nh_min`** is impossible by construction when seeds exist, but the **`max`** keeps the definition consistent.

### 4.2 Deduplication against blocks

Let **`block_nh_set`** be the set of all **`nh`** for which **`block_truncated`** rows were (or would be) produced in this chunk, **including** values reached **only** by the tower.

```text
slide_nh_list = [ wh for wh in candidates if wh ∉ block_nh_set ]
```

**Concrete rule:** if **`N`** (e.g. **120**) is **already** in **`block_nh_set`**, it is **removed** from sliding — **no second policy layer**. If **`N`** is **not** in **`block_nh_set`** and **`N`** appears in **`candidates`**, sliding **does** run at **`wh = N`**.

### 4.3 Placements (origins)

**Sliding** uses **valid-box** averages: the window must lie **inside** the domain. This is **separate** from block tiling (§2.2). By default, the implementation uses **few** placements per axis (e.g. **`sliding_outputs_h = 2`**) when stride is not explicit — **not** a dense stride-1 grid. “Corners” in **sliding** refers to **which valid origins** are sampled (e.g. flush to low and high valid positions), **not** to the block **index-1 truncation** rule.

### 4.4 Arrow metadata

**`reduction_kind`:** **`sliding_valid`** (sliding-only mode) or **`hybrid_sliding`** (after the block pass in hybrid).

---

## 5. Vertical coarsening (`fz`)

**Goal:** mirror §3–§4 **structure** for the column: **(i)** block + prime tower from seeds in the **lower half** of the allowed **`fz` range**, **(ii)** in **hybrid**, sliding window **heights** subsampled from the **upper half**, minus duplicates already produced as blocks. **Physics** is **not** symmetric with horizontal: there is **no** `min_h`-style **minimum** vertical coarsening — the finest scale is always **`fz = 1`**. The **only** hard vertical constraint in this spec is **`max_dz`** (plus **`nz`**).

**Symbols** (see §2.1): **`fz_cap`** from **`max_dz`**, **`fz_max = min(nz, fz_cap)`**, **`fz_half = ⌊fz_max/2⌋`**.

**Narrative example (your numbers):** mean layer thickness **`dz_ref = 10` m**, **`max_dz = 400` m** → **`fz_cap = 40`** (“at most 40 native layers per coarse layer” if the column were deep enough). On a real chunk, **`fz_max ≤ min(40, nz)`**. Then **`fz_half = 20`**: **tower seeds** subsample **`[1, 20]`**; **hybrid sliding** subsamples **`wz`** from **`[21, fz_max]`** (i.e. up to **40** when **`nz ≥ 40`**).

### 5.1 Lower band — block reductions and vertical tower

**Seeds**

```text
fz_seeds = subsample_closed_range(1, fz_half, B)   if fz_half ≥ 1
         = subsample_closed_range(1, fz_max, B)    if fz_half < 1
```

These are **entry points only**; the tower may reach many **`fz`** values not in **`fz_seeds`**.

**Tower (prime 1D pools, DFS, deduplication)**

Mirror §3.2, but:

- **Atomic move:** a **prime** **`p`** applies a **non-overlapping `p`-layer mean along `z`** on the **current** vertical grid, with **§2.3** truncation (low-`k` origin, drop incomplete tail).
- **Native-equivalent factor:** **`fz_child = fz · p`**.
- **Validity** of **`p`** at a node with native-equivalent **`fz`** and **current** vertical extent **`nz_c` coarse layers** (after prior truncation):

  1. **`fz_child ≤ fz_max`**, and  
  2. **`⌊nz_c / p⌋ ≥ 1`** (at least one full **`p`**-layer tile from **`k = 1`**).

**Why primes only:** same rationale as §3.2 — composites are sequences of prime steps on successively coarser **`z`** grids.

**Deduplication:** collect distinct **`fz`** across all seeds and DFS branches (same **`fz`** must not imply duplicate block work).

**Combined with horizontal:** for **each** pair **`(nh, fz)`** in the Cartesian product of the horizontal block schedule (§3) and the vertical **`fz`** set from this subsection, apply horizontal coarsening at **`nh`**, then vertical coarsening at **`fz`** (implementation may cache by **`nh`** and/or reuse vertical intermediates — not normative here).

### 5.2 Hybrid — vertical sliding extras (upper band)

Let **`block_fz_set`** be every **`fz`** emitted (or that would be emitted) from the **block** pass for this chunk, **including** values reached **only** via the vertical tower.

```text
fz_slide_lo = fz_half + 1
fz_slide_hi = fz_max
candidates_z = subsample_closed_range(fz_slide_lo, fz_slide_hi, B)
slide_fz_list = [ wz for wz in candidates_z if wz ∉ block_fz_set ]
```

**Rationale:** same as §4.1 for **`wh`**: sliding fills **upper** half of the **`fz`** ladder that blocks might miss or under-sample; **dedupe** so block and sliding do not double-emit the same vertical scale.

**Implementation allowance:** if **`fz_slide_lo > fz_slide_hi`** (e.g. **`fz_max = 1`**) or **`slide_fz_list`** is empty, implementations **may** fall back to **`subsample_closed_range(1, fz_max, B) \\ block_fz_set`** so **hybrid horizontal** sliding (§4) still pairs with **some** vertical **`wz`** where needed. Document any fallback in code comments / changelog.

### 5.3 Sliding-only mode (vertical)

Mirror horizontal sliding-only: subsample up to **`B`** values of **`fz`** (window height **`wz`**) in **`[1, fz_max]`** using **`subsample_closed_range`**. No divisor requirement; valid-box **`z`** windows obey §2.3-style **containment** inside **`1:nz`**, not the block truncation formula.

### 5.4 Arrow fields and thickness

- **`reduction_fz`:** block **`fz`** or sliding **`wz`** as above.  
- **`resolution_z`:** physical thickness of that coarse vertical layer = **sum of `dz_native_profile` over the native levels** that the mean aggregates (block path: §2.3 grouping; sliding: the window’s native span).

**Truncation example:** **`nz = 17`**, **`fz = 5`** → **`nz_used = 15`**, **3** coarse levels from **`k = 1:15`**; native **`k = 16, 17`** omitted for that scale.

---

## 6. Modes (summary)

| Mode | Horizontal | Vertical |
|------|------------|----------|
| **`block`** | §3 seeds + tower. Optional override: explicit `block_triples` in `spatial_info`. | §5.1 seeds + vertical tower; **`fz_max`** from §2.1. |
| **`sliding`** | §4 subsampled **`wh`**, deduped vs blocks if ever combined. | §5.3 subsampled **`wz ∈ [1, fz_max]`**. |
| **`hybrid`** | §3 then §4; **`slide_nh_list`** from §4.2. | Block pass §5.1 then **`slide_fz_list`** from §5.2 (with §5.2 fallback if needed). |

---

## 7. Duplicate scale across block and sliding

- **Horizontal:** if **`wh = nh`** for some block-produced **`nh`**, **do not** emit redundant sliding rows for that **`wh`** — **block wins** (§4.2).  
- **Vertical:** if **`wz = fz`** for some block-produced **`fz`**, **do not** emit redundant hybrid sliding rows for that **`wz`** — **block wins** (§5.2).

---

## 8. What a newcomer will see in `.arrow` (qualitative)

- **`resolution_h`:** **`nh · dx_native`** (or sliding **`wh · dx_native`**).  
- **`resolution_z` / thickness:** from **`dz_native_profile`** over the levels in each coarse cell (§5.4).  
- **Block path:** **many** distinct **`nh`** and **`fz`** from the two towers — **not** only **`B`** seeds per axis.  
- **Sliding path:** at most **`B`** candidate **`wh`** after §4.2 and **`B`** candidate **`wz`** after §5.2.  
- **`reduction_kind`**, **`reduction_nh`**, **`reduction_fz`**, **`truncation_*`:** identify how each row was produced.

Exact counts depend on **`N`**, **`nz`**, **`fz_cap`**, **`nh_min`**, **`B`**, primes, truncation, and cloud masks.

---

## 9. Non-goals and implementation status

- **Normative:** prime-only **atomic** steps in **both** towers; **all** valid primes in range considered (§2.5).
- **Not normative:** hard cap on distinct **`nh`** or **`fz`** — if added later, a **performance** knob, documented separately.
- **Approval gate:** treat this file as **stakeholder-approved** before changing production schedules in code; until then, runtime behavior may **diverge** (e.g. divisor-only **`fz`**, incomplete horizontal DFS). This document is the **single normative target** for those schedules.

---

## 10. Design decisions (defaults captured here)

| Topic | Decision in this spec |
|-------|------------------------|
| Seed range | **`[nh_min, nh_half]`**, **`B`** subsamples. |
| Tower reach | **`nh`** may go **up to `N`** with blocks. |
| Atomic pool factors | **Primes only**; composites as **sequences** of primes. |
| Sliding candidates | **`[max(nh_min, nh_half), N]`**, **`B`** subsamples, minus **`block_nh_set`**. |
| **`nh = N`** in sliding | **Emit** if **`N ∉ block_nh_set`** and **`N`** is in the subsample; **omit** if already a block scale. |
| Block vs sliding duplicate | **Block wins**; sliding list filtered (§4.2). |
| Horizontal block tiling | **Low index `1`** on **`x,y`**, full tiles, tail **dropped** (§2.2). |
| Vertical block tiling | **Low index `1`** on **`z`**, full **`fz`**-groups, tail **dropped** (§2.3); **no** **`fz | nz`** requirement. |
| Vertical physics | **Cap** via **`max_dz`** (**`fz_cap`**); **no** `min_h`-analog; finest **`fz = 1`**. |
| **`fz_max`** | **`min(nz, fz_cap)`** with **`fz_cap`** from **`fz · dz_ref ≤ max_dz`** (§2.1). |
| Vertical lower band | **`B`** seeds in **`[1, fz_half]`** + **prime 1D** **`z`** tower (§5.1). |
| Vertical upper band (hybrid) | **`B`** subsamples in **`[fz_half+1, fz_max]`**, minus **`block_fz_set`**; optional fallback (§5.2). |

**Status:** *Plan / normative spec — **approve before implementation**; then code and tests should match §2–§5 and this table.*
