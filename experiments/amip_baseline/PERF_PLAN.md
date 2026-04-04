# GoogleLES Performance Plan

## Goals
- Measure where time and allocations are spent without JIT noise.
- Optimize one stage at a time with before/after metrics.
- Keep a reproducible command for regression checks.

## Baseline Command
Preferred REPL workflow:

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")
```

Terminal fallback:

```bash
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")'
```

Optional knobs:

```bash
SITE_ID=343 MONTH=1 PROFILE_TIMESTEPS=8 PROFILE_REPEATS=5 julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")'
```

## Stages Tracked
1. Metadata load (`load_zarr_simulation`)
2. Cache load (`_load_googleles_cache`) -> includes network + decompress
3. In-memory single timestep processing
4. In-memory all cached timesteps processing

Each stage reports:
- min/median/mean time
- min/median allocated MB
- mean GC seconds
- Ideally we'd run a full profiler so we'd know where the contribution of every line of code without having to manually add timing blocks.

## Optimization Order
1. Cache load path
   - Reduce extra copies and conversions.
   - Verify `Float32` assumptions.
2. In-memory timestep path
   - Reduce DataFrame and Dict allocation churn in `DatasetBuilder.process_abstract_chunk`.
   - Reuse buffers where possible.
3. Serialization path
   - Measure `Arrow.write` separately once processing is stable.

## Regression Discipline
After each optimization patch:
1. Run the baseline command.
2. Record metrics in a commit message or issue note.
3. Do not keep changes that regress both time and allocations.
4. We will add tests to the test suite to ensure type stability and allocation-free (or minimality)

## About `stream not initialized`
This error is from background HTTP/OpenSSL idle-connection monitoring tasks.
It is noise around the network client lifecycle and not the core timestep math kernel.
We must find a way to fix this so it doesnt dump into output for the user

## TODO Handoff (Pick Up Later)

### TODO 1: Replace `findall` + CartesianIndex gather with one-pass column fill
Status: todo
Scope:
- `DatasetBuilder.flatten_and_filter!`
- `DatasetBuilder._gather_3d`, `_gather_profile`, `_gather_3d_sum`

Problem:
- Current path allocates `valid_indices = findall(!, mask)` and then gathers each column via repeated indexed passes.
- This creates extra index-vector churn and repeated cache-unfriendly scans.

Plan:
1. Count valid cells with one mask pass.
2. Preallocate all output vectors once at `n_valid`.
3. Fill all columns in one traversal over mask and source arrays.
4. Keep schema ordering exactly unchanged.

Validation:
- Run `test/test_dataset_builder.jl`.
- Run chunk profiler and compare alloc report lines in `dataset_builder.jl`.

### TODO 2: Add scratch-buffer coarse-graining API and reuse across levels
Status: todo
Scope:
- `CoarseGraining.cg_2x2_horizontal!` (new)
- Optional: `CoarseGraining.compute_covariance!` (new)
- `DatasetBuilder.process_abstract_chunk`

Problem:
- Even after direct start-factor reduction, each additional level allocates fresh coarse arrays.

Plan:
1. Add in-place coarse-grain kernel writing into destination arrays.
2. Use ping-pong buffers (`A` -> `B` -> `A`) for each field family.
3. Preserve numerical results and shape truncation semantics.

Validation:
- Run existing tests.
- Profile chunk kernel and verify lower alloc samples on `coarse_graining.jl` lines.

### TODO 3: Pre-allocate and reuse mask/nonfinite buffers in chunk loop
Status: todo
Scope:
- `DatasetBuilder.process_abstract_chunk`

Problem:
- `sparse_mask`, `nonfinite_mask`, and `combined_mask` are reallocated each emitted level.

Plan:
1. Allocate max-size buffers once for current coarse shape.
2. Reuse buffers with `fill!` each iteration.
3. Reallocate only when shape changes after next coarse step.

Validation:
- Tests unchanged.
- Chunk alloc profile should show reduced mask-related allocations.

### TODO 4: Add `PROFILE_SITE` list sweep for non-empty-cloud chunk selection
Status: todo
Scope:
- `experiments/amip_baseline/profile_googleles_pipeline.jl`

Problem:
- Some sites/months have fully masked early timesteps; this causes profiling noise or synthetic fallback.

Plan:
1. Add optional env var for candidate sites (comma-separated).
2. Probe first few timesteps and choose first non-empty candidate automatically.
3. Print selected site/timestep in output header.

Validation:
- Run chunk profile with default settings and confirm real-case path is used.

### TODO 5: End-to-end comparison script for before/after report diff
Status: todo
Scope:
- `experiments/amip_baseline/` (new helper script is acceptable)

Problem:
- Manual report inspection is error-prone and slow.

Plan:
1. Parse `*_time_flat.txt` and `*_allocs_flat.txt` for project-local rows.
2. Emit table with `line`, `before_count`, `after_count`, and delta.
3. Store output in report directory.

Validation:
- Run against two profile directories and verify deterministic output.

## Resume Commands

Run unit checks:

```bash
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/test/test_dataset_builder.jl")'
```

Run chunk profile (saved to timestamped folder):

```bash
cd /home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline && \
PROFILE_SCOPE=chunk PROFILE_TIMESTEPS=8 PROFILE_CHUNK_REPEATS=20 PROFILE_SAMPLE_RATE=0.05 PROFILE_MINCOUNT=5 \
julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline \
   -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")'
```