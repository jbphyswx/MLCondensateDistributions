# V2 Remote Load Reduction Questions

This note tracks the open questions around reducing remote zarr reads in the GoogleLES v2 pipeline. The goal is to minimize bandwidth and decompression work, not just to optimize the local compute kernel.

**Canonical axis / chunk semantics (do not guess from `_ARRAY_DIMENSIONS` list order alone):** [googleles_zarr_layout.md](googleles_zarr_layout.md). That doc explains why `chunks[i]` aligns with `size[i]` in Julia but **not** with `_ARRAY_DIMENSIONS[i]` for these stores.

## Current Observations

- v2 processing is faster than legacy when data is already in memory, but remote zarr reads still dominate end-to-end behavior.
- The GoogleLES zarr arrays expose chunk metadata of `(60, 124, 124, 1)` for arrays shaped `(480, 124, 124, 73)`.
- The 480-length axis is z and the 73-length axis is t (see layout doc for the full `size` ↔ `_ARRAY_DIMENSIONS` mapping).
- This means z is chunked in blocks of 60 layers, while t is chunked at single timesteps.
- `_has_cloud_after_2x2` currently checks cloud presence at any z, but given our deterministic z reductions, we should be able to avoid loading many z layers.
- Many profiles look mostly dry, with isolated cloudy regions near the inversion.

## Main Hypothesis

We may be able to avoid loading large fractions of the remote data by exploiting two facts:

1. Horizontal and vertical reductions are deterministic.
2. Dry regions can often be ruled out early and never contribute to any future coarse level.

If that is true, the pipeline should request only the subdomains that can still affect a surviving coarse cell.

## Questions To Answer

### 1. Can we skip whole z spans that cannot contribute to any future reduction?

- Given the deterministic vertical coarsening schedule, can we identify vertical intervals that will never affect any emitted level?
- Can we drop those intervals before loading them from remote storage?
- How much does the answer depend on whether cloudy layers are contiguous or separated by dry gaps?

### 2. Can we request contiguous z domains only?

- If cloudy layers are separated by large dry gaps, can we load only the contiguous cloudy ranges plus whatever padding is needed for future coarsening?
- What is the minimum safe padding so later reductions do not incorrectly treat separated cloudy slabs as adjacent?
- Do we need to preserve gap structure explicitly in the representation?

### 3. How much can be skipped for non-qc variables?

- Since `q_c` is used as the early cloud filter, can we load `q_c` first and only fetch the other fields for surviving vertical ranges?
- Could we avoid loading 4 of the 8 vertical chunks in typical mostly-dry cases?
- Is there a good rule for loading `ta`, `hus`, `ua`, `va`, etc. only where `q_c` says the region might matter?


### 4. What is the granularity of useful requests below the chunk level?

- If requests smaller than one zarr chunk continue to get faster, we may need a more precise fetch strategy.
- Do we get additional speedup when reading sub-chunk, or do the zarr materialization methods always load full chunks?
- Does requesting smaller windows inside a chunk still reduce wall time, or do we hit chunk-level decompression cost anyway?
- Should we test whether the storage backend gives any benefit when the requested subrange is much smaller than the chunk size?

### 5. Can the pipeline generate contiguous z domains before loading everything else?

- Can we first scan `q_c` at a coarse or cheap stage, then derive the set of vertical spans that survive all future reductions?
- Once those spans are known, can we load the remaining variables only for those spans?
- Is it cheaper to do a lightweight pass on `q_c` first and a second pass on the other fields?

## Candidate Experiments

1. Measure time for `q_c`-only loads at increasing vertical window sizes.
2. Measure time for full-field loads on the same windows.
3. Compare one contiguous window versus several smaller windows covering the same total z extent.
4. Test whether loading below a chunk boundary changes runtime at all.
5. Prototype a two-pass strategy:
   - pass 1: load `q_c` only
   - pass 2: load remaining fields only for surviving z spans

## Findings So Far

### A. Full-field load scaling over timesteps (9 fields, materialized)

From `experiments/amip_baseline/profile/profile_zarr_load_scaling.jl`:

- t=1: total 2.318 s (load 0.157 s + materialize 2.161 s)
- t=2: total 1.904 s
- t=4: total 5.327 s
- t=8: total 8.702 s
- t=16: total 20.835 s
- t=32: total 43.305 s
- t=64: total 84.781 s
- t=73: total 83.488 s

Interpretation:

- For larger t, cost is roughly linear in timesteps.
- The materialization stage dominates; the initial "load" is largely lazy view setup.
- The 64 vs 73 inversion is noise/cache effect, not a true speedup.

### B. q_c-only z-window scaling (single timestep)

From `experiments/amip_baseline/profile/profile_qc_z_window_scaling.jl` (corrected axis mapping):

- shape=(480, 124, 124, 73), z_chunk=60
- initial run showed z=1: 0.799 s (cold-start outlier)

Warmup-per-window rerun (`WARMUP_PER_Z=true`):

- z=1: 0.043 s
- z=2: 0.005 s
- z=4: 0.011 s
- z=8: 0.010 s

---

## CRITICAL LESSON: Remote Data Gridlock (2026-04-04)

### The Problem

After implementing span-wise z-level filtering to reduce allocations, v2 experienced a **10-100x gridlock regression**. The pipeline would timeout or extremely slow during remote data processing, with obvious thrashing during Zarr decompression phases. Initial diagnosis blamed "per-span remote reads," suggesting reads were happening too frequently.

### The Actual Root Cause (User Correction)

The gridlock was **NOT** caused by frequent reads. It was caused by **computing on lazy remote-backed Zarr array views instead of materializing data locally first**.

The pipeline was:
1. ✗ Reading a z-span from remote (fast)
2. ✗ Using that remote view directly in computations (catastrophic - each array operation involved remote decompression + HTTP round-trip)
3. ✗ Multiple downstream functions computing on the same lazy view = repeated decompression

The fix was:
1. ✓ Read a z-span from remote
2. ✓ **Materialize immediately into a preallocated local buffer** using `copyto!()`
3. ✓ Pass local buffer views to downstream functions

### Why This Matters

**Never compute on lazy remote-backed arrays. Always materialize first.**

Zarr and remote HTTP storage are designed for fast sequential reads into local memory. They are catastrophically slow for random-access computation. A single `tanh()` call on a lazy Zarr view can trigger decompression requests for multiple chunks.

### Implementation: `_load_googleles_timestep_fields_into!`

**Canonical algorithm and file map:** see [googleles_build_tabular.md](googleles_build_tabular.md) (the package does **not** load `utils/__deprecated__/`).

The loader must **not** pass lazy Zarr views to `process_abstract_chunk`. The actual implementation (in `utils/build_training_common.jl`) fills a **reused** `scratch` dict with views, then `copyto!` into preallocated 3D buffers for **only** the requested native `z_range`:

```julia
function _load_googleles_timestep_fields_into!(dest, ds, timestep_idx; field_specs, z_range, scratch)
    empty!(scratch)
    _load_googleles_timestep_fields!(scratch, ds, timestep_idx; field_specs=field_specs, z_range=z_range)
    for (_, c_var) in field_specs
        copyto!(@view(dest[c_var][:, :, z_range]), scratch[c_var])
    end
    return dest
end
```

This ensures:

- Only the contiguous native slab `z_range` is read for each non-`q_c` field (plus whatever full Zarr chunks the store must decode).
- Materialization happens immediately (`copyto!`); downstream code uses `@view dest[c_var][:, :, z_range]` into **local** `Array{Float32,3}` memory.
- `scratch` is reused across spans to avoid allocating a new `Dict` per span.

### Code Pattern (How v2 Now Works)

**Full case path (per cloudy timestep):** one `q_c` materialization per timestep; **per contiguous run** of `z_keep_mask`, one call to `_load_googleles_timestep_fields_into!` for non-`q_c` fields (minimal z extent for that run, not necessarily full `nz`).

```julia
non_qc_zarr_scratch = Dict{String, AbstractArray{Float32, 3}}()
# ... q_c_buf filled once per timestep; z_keep_mask built from full-column z_future_factors ...

_foreach_true_span(z_keep_mask) do k_start, k_end
    z_range = k_start:k_end
    _load_googleles_timestep_fields_into!(
        non_qc_buffers, ds, local_t;
        field_specs=non_qc_specs, z_range=z_range, scratch=non_qc_zarr_scratch,
    )
    ta = @view non_qc_buffers["ta"][:, :, z_range]
    q_c_span = @view q_c_buf[:, :, z_range]
    # ... partition_condensate, process_abstract_chunk — all local ...
end
```

### Generalized Rule

For any remote/lazy data pipeline:

1. **Identify the minimal read footprint** (z-span, horizontal subset, etc.)
2. **Read and materialize that footprint into local working memory**
3. **Never pass lazy remote views downstream**
4. **Compute exclusively on materialized buffers**

If downstream code ever receives a lazy view and computes on it, thrashing will result.

### Testing for This Problem

If you suspect lazy-view computation gridlock:
- Monitor system load during computation (high load ≠ responsive computation)
- Check Zarr HTTP request logs (many sequential small requests indicate re-reading)
- Look for timeouts in decompression phases (not in read phases)
- Profile with `@time` on individual operations; they'll show huge latency

**Fix:** Materialize suspect data into local buffers before passing downstream.
- z=16: 0.005 s
- z=32: 0.005 s
- z=64: 0.018 s
- z=128: 0.011 s
- z=256: 0.048 s
- z=320: 0.221 s
- z=480: 0.027 s

Newer cache-resistant check (distinct timestep per z-window in one process):

- z=1 (t=2): 0.025 s
- z=2 (t=3): 0.127 s
- z=4 (t=4): 0.114 s
- z=8 (t=5): 0.052 s
- z=16 (t=6): 0.066 s
- z=32 (t=7): 0.058 s
- z=64 (t=8): 0.074 s
- z=128 (t=9): 0.113 s
- z=256 (t=10): 0.121 s
- z=320 (t=11): 0.070 s
- z=480 (t=12): 0.083 s

Interpretation:

- These same-process runs are noisy and cache biased, but confirm the benchmark is now sweeping true z.
- The mismatch between the two runs is expected under cache reuse and transport jitter.
- Results should not be interpreted as final z-scaling laws until cold-process benchmarking is done.
- This is q_c-only and cannot be compared directly to the 9-field timestep benchmark.
- The original z=1 spike was dominated by first-touch overhead (cold fetch/decompression and startup effects).

### C. q_c-only z-window scaling (cold-process baseline, decision-grade)

From `experiments/amip_baseline/profile/profile_qc_z_window_scaling_cold.jl` with `REPEATS=3`:

- z=1: 0.088957 s
- z=2: 0.132376 s
- z=4: 0.083773 s
- z=8: 0.089288 s
- z=16: 0.138831 s
- z=32: 0.090164 s
- z=64: 0.136207 s
- z=128: 0.144234 s
- z=256: 0.286141 s
- z=320: 0.156612 s
- z=480: 0.165759 s

Interpretation:

- Cold-process medians are far more stable than same-process microbenchmarks.
- Timing still is not strictly monotonic with z due to object-store/network variance.
- However, this is now a usable baseline for comparing alternative access strategies.
- Next comparison must be like-for-like: run the same cold-process harness for all 9 fields.

## Data Quality Caveat

Current microbenchmarks are contaminated by caching/warm state because multiple sizes are tested in one Julia process. For decision-grade numbers, use cold-process measurements per data point.

Additional check (distinct timestep per z-window in one process) still showed non-monotonic timings, which is consistent with cache reuse and transport jitter. This confirms we should not infer final scaling laws from same-process microbenchmarks.

## What We Can Answer Now

1. Is `z=1` around 0.8 s a real scaling point?
   - No. It is a cold-start outlier (first-touch overhead).
2. Is the current z-window benchmark enough to conclude exact z scaling?
   - No. Same-process runs are too cache/jitter sensitive.
3. Is the axis mapping now resolved?
   - Yes: shape `(480, 124, 124, 73)` with z on the 480 axis and t on the 73 axis.
4. Is the project direction still valid?
   - Yes: q_c-first selective loading remains the best candidate path, but it needs clean evidence from cold-process measurements.

## Updated Execution Plan

1. Build a cold-process harness (one fresh Julia process per data point). (done)
2. Re-run q_c-only z-window scaling with medians and spread statistics. (done)
3. Add equivalent 9-field z-window scaling.
4. Compare contiguous vs disjoint z-window reads at fixed total layers.
5. Use those results to decide on implementing q_c-first selective loading in v2.

## What To Keep In Mind

- The current chunking appears to be z-heavy in the first axis and single-timestep in the last axis, so the practical bandwidth question may depend on which axis we are trimming.
- Any filtering logic must preserve the deterministic reduction tree; otherwise we risk changing the emitted dataset.
- The main objective is not just lower allocations, but less remote data fetched per timestep.

## Working Assumption

If the data is 99% dry, the best path forward is probably a staging strategy that:

1. reads a cheap cloud indicator first,
2. computes the minimal set of surviving z regions,
3. loads only those regions for the expensive variables,
4. and keeps the reduction structure exact.

## Next Step

Run this sequence in order:

1. Add cold-process benchmark mode (fresh Julia process per point) for z-window tests.
2. Repeat `q_c`-only z-window scaling under cold-process mode.
3. Add and run 9-field z-window scaling on the same z windows.
4. Compare one contiguous z window vs multiple disjoint windows with equal total layers.
5. Decide whether to implement q_c-first selective loading in v2 orchestration.

