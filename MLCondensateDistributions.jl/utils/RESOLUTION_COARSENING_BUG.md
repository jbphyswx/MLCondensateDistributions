# Resolution Coarsening Bug Report and Fix Plan

## Context

Observed in `run_resolution_impact_covariances!()` outputs and in generated Arrow data:

- Horizontal resolutions in outputs top out near ~3122 m (for the sample GoogleLES file), not ~6000 m.
- Resolution coverage is sparse and depends on binary factors.
- Plot confusion was amplified by a separate heatmap axis-orientation bug (already patched in plotting code).

## What Is Happening

For GoogleLES example:

- `x` has 124 points from 0 to 6000.
- Native spacing is `dx_native = 6000 / (124 - 1) = 48.780487...` m.

Current horizontal coarsening in `process_abstract_chunk` (`utils/dataset_builder.jl`) does:

1. Start at a power-of-two factor large enough to pass `min_dh` (default 1000 m).
2. Repeatedly apply 2x pooling only.
3. Stop when horizontal coarse grid cannot continue (`min(size(c_qt,1), size(c_qt,2)) < 2`).

Because pooling functions use integer division (`div(nx, factor)`), non-divisible remainder cells are dropped.

For `nx = 124`:

- Start factor becomes `32` (since `48.78 * 32 = 1560.98`).
- Next binary factor is `64` (`3121.95`).
- Next would be `128`, but `div(124, 128) = 0` (invalid), and the loop stops before that level.

So data never reaches a true full-domain horizontal aggregate (~6000 m).

## Root Cause

### Algorithmic bug (data generation)

The current scheme assumes binary factors are sufficient to represent desired scales, but with non-power-of-two grid sizes they are not.

Key issues:

- Horizontal factors are restricted to powers of two.
- Coarsening drops remainder cells (`div`) instead of handling non-divisible extents.
- There is no explicit guaranteed full-domain output level.

### Visualization confusion (already patched)

A separate heatmap axis/matrix orientation issue caused misleading plots (apparent emptiness/continuous gradients at wrong axis scales). That issue is independent of the data-generation bug above.

## Why ~6000 m Is Missing

`domain_h` is present as metadata, but `resolution_h` levels are generated only through pooling factors.

If factor generation cannot produce `factor ~= 124` (or an explicit full-domain aggregation), `resolution_h ~= 6000` is never emitted.

## Correct Behavior (Desired)

1. Always include full-domain horizontal aggregate (1x1 horizontal) for every emitted z-scheme/time step.
2. Preserve binary levels for speed, but allow additional factors (e.g., 3, 5, 7 or user-specified).
3. Define explicit policy for non-divisible factors:
   - either weighted boundary blocks,
   - or deterministic crop/pad strategy with clear metadata.
4. Keep `resolution_h` values tied to actual aggregation geometry, not implied binary-only progression.

## Fix Plan

### Phase 1 (must-have correctness)

1. Add explicit full-domain emission path in `process_abstract_chunk`:
   - compute horizontal means/products directly over full `(x,y)` plane,
   - emit rows with `resolution_h = domain_h` (or exact extent),
   - do this even when binary ladder cannot reach that factor.

2. Add metadata columns for auditability (optional but recommended):
   - `h_factor_x`, `h_factor_y`,
   - `h_policy` (`binary_div`, `full_domain`, `crop`, etc.),
   - `effective_nx`, `effective_ny` used in aggregation.

### Phase 2 (factor flexibility)

Support configurable factor families:

- default: binary ladder + full-domain,
- optional: include custom factors (`[2,3,5,7,...]` multipliers or explicit list).

Implementation direction:

- Build factor list first, then evaluate each factor independently.
- Do not rely on repeated 2x from prior level as the only route.

### Phase 3 (non-divisible handling policy)

Implement one explicit policy (and document it):

- `crop` (current behavior but explicit),
- `pad`, or
- `weighted_edge_blocks` (preferred for physical consistency).

## Performance Notes

Current runtime is high because coarse-graining is repeated for many fields and moments at many levels.

High-impact opportunities:

1. Compute reusable block sums (summed-area / integral image approach) for means and products.
2. Batch operations across fields to reduce repeated memory passes.
3. Reuse preallocated buffers across levels.
4. Parallelize across z-levels/time chunks where safe.
5. Delay filtering/masking until after aggregation where possible to reduce branch-heavy inner loops.

## Validation Checklist After Fix

1. For sample GoogleLES file (`nx=124`, extent 6000):
   - `resolution_h` includes approximately `1560.98`, `3121.95`, and `6000`.
2. `max(resolution_h)` in Arrow outputs reaches domain scale.
3. Plots show expected discrete x bins only (no fake continuous x scale).
4. Unit test confirms full-domain level exists regardless of divisibility by powers of two.

## Suggested Unit Test

Add a test in `test/test_dataset_builder.jl`:

- build from a synthetic 124x124 field with known domain extent,
- run chunk processing,
- assert at least one output row has `resolution_h ≈ domain_h` within tolerance.

---

This is a real data-generation bug (not just a plotting artifact): the current binary-only, divisible-block ladder does not guarantee domain-scale outputs for non-power-of-two grids.
