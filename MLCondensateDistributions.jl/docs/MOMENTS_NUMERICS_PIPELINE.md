# Moments, variance, and covariance numerics (tabular builder)

This document is the **design reference** for how MLCondensateDistributions computes second moments that end up in Arrow training rows (`var_*`, `cov_*`, `tke`). It complements [`COARSENING_REDUCTION_IMPLEMENTATION_PLAN.md`](COARSENING_REDUCTION_IMPLEMENTATION_PLAN.md).

## Schema contract (Arrow)

The canonical training schema ([`dataset_spec.md`](../dataset_spec.md), [`utils/dataset_builder.jl`](../utils/dataset_builder.jl)) exposes **derived** columns only: means, `tke`, `var_*`, `cov_*`, and metadata. It does **not** require persisting raw second moments such as `⟨x²⟩` or `⟨xy⟩` as named features. Those quantities may appear only as **internal** arrays while building a row.

Downstream Python ([`python/dataset.py`](../python/dataset.py)) loads arbitrary `feature_cols` / `target_cols` as `float32`; it does not assume a specific variance definition beyond what is stored in the table.

## Definitions (shipped behavior)

For a coarse voxel that aggregates **n** native samples (non-overlapping block):

- **Population variance** (per voxel): `Var(x) = (1/n) Σᵢ (xᵢ − x̄)²` with **x̄** the block mean. Implemented as **`M2 / n`** where **`M2 = Σᵢ (xᵢ − x̄)²`** (sum of squared deviations for that block).
- **Population covariance** (per voxel): `Cov(x,y) = (1/n) Σᵢ (xᵢ − x̄)(yᵢ − ȳ)`. Implemented as **`C / n`** where **`C = Σᵢ (xᵢ − x̄)(yᵢ − ȳ)`** using the block means **x̄, ȳ**.

**TKE** uses `tke = (1/2) * (Var(u) + Var(v) + Var(w))` with the same population variance convention on the velocity components.

Sliding valid-box reductions (sparse convolutional windows) still form **mean ⟨x⟩**, **mean ⟨x²⟩**, **mean ⟨xy⟩** over the window, then use `⟨x²⟩ − ⟨x⟩²` and `⟨xy⟩ − ⟨x⟩⟨y⟩` at emit time. That path matches the old algebra; numerics are helped primarily by **wider accumulators** in the inner loops (Phase 1).

## Failure modes (why this matters)

1. **Catastrophic cancellation:** `⟨x²⟩ − ⟨x⟩²` when `|x|` is large and variance is tiny — operands are both huge and nearly equal; the difference loses almost all bits in `Float32`.
2. **Loss in `Σ x²`:** For large **n**, `Σ x²` is dominated by `n x̄²`; the **`n·Var`** contribution can sit below one ulp of the sum in `Float32`, so `⟨x²⟩` never encodes the variance.
3. **Wrong hierarchical algebra:** Averaging child **`⟨x²⟩`** fields (or child **`⟨xy⟩`**) then subtracting is **not** the same as merging sufficient statistics for nested blocks in floating point. **Chan/Welford-style merge** of **`(n, x̄, M2)`** (and Pebay-style merge for **`C`**) matches exact nested pooling in ℝ and stays stable when child summaries are already aggregated.

## Welford, Chan, and “Pebay” (not the same thing)

- **Welford** usually means the **online** one-pass update for running mean and sum of squared deviations (M2). **Chan** (or parallel **Welford batch merge**) is the **algebra for combining two already-summarized groups** \((n_1,\mu_1,M_{2,1})\) and \((n_2,\mu_2,M_{2,2})\) into one — what `merge_variance_chan` implements. Same sufficient statistics; one is streaming, one is merge.
- **Pebay** refers to **Philippe Pebay** (Sandia / parallel statistics literature). His work gives **parallel merge formulas for covariances** (and higher moments). The scalar merge we use for **`C`** (sum of \((x-\bar x)(y-\bar y)\) within a block) is the **pairwise covariance case** of that family — analogous role to Chan for variance, but for **two** variables. It is **not** identical to univariate Welford; it is the correct merge for **`C`**, just as Chan is the correct merge for **`M2`**.

### Phase 1 — Wider reduction accumulators

In block and valid-box kernels, **scalar accumulators** for sums and product-sums use **`Float64`** when the array eltype is **`Float32`**, then cast **once** per output voxel to `T`. No extra arrays; only a few scalars per inner loop.

**FMA** (fused multiply-add): `fma(a,b,c)` does `a*b+c` with one rounding. Optional future micro-optimization for product sums; orthogonal to widening.

### Phase 2 — Block path: central sums + merge

- **Native / first reduction:** For each output voxel, compute **x̄** and **M2** (or **x̄, ȳ, C**) over the block using **widened scalar accumulators** when `eltype` is `Float32` (same rule as Phase 1), then store **`Float32`** fields (two-pass block kernels in [`array_utils.jl`](../utils/array_utils.jl)).
- **Hierarchical merge scratch:** Child **μ**, **M2**, and **C** are stored as `T`, but **small temporary buffers** inside Chan/Pebay merge use **`ArrayUtils._block_reduction_accum_type(T)`** (`Float64` when `T === Float32`). That is **not** persisting Float64 in Arrow — it avoids doing the merge entirely in `Float32`, which would reintroduce loss when combining many coarse cells. **`Vector{T}` scratch would be the wrong default** for `T === Float32` here: the merge algebra needs extra precision even though inputs and outputs remain Float32.
- **Horizontal / vertical chaining:** Instead of mean-pooling **`⟨x²⟩`** or **`⟨xy⟩`**, **merge** child **`(n, x̄, M2)`** or **`(n, x̄, ȳ, C)`** with Chan (variance) and Pebay-style (covariance); see Pebay (2008), *Formulas for parallel computation of covariances*.

Merge helpers and covariance field utilities live in [`utils/statistical_methods/StatisticalMethods.jl`](../utils/statistical_methods/StatisticalMethods.jl) (exported as `StatisticalMethods`; import as `SM` with `using MLCondensateDistributions: StatisticalMethods as SM` if desired). Block-tower wiring is in [`utils/coarsening_pipeline.jl`](../utils/coarsening_pipeline.jl) and [`utils/dataset_builder_impl.jl`](../utils/dataset_builder_impl.jl).

**Emit (sliding / mean–product path):** [`StatisticalMethods.covariance_from_moments!`](../utils/statistical_methods/covariance_fields.jl) (`⟨xy⟩−⟨x⟩⟨y⟩` in place). **TKE fields** from velocity first/second means or from **sums of squared deviations** (often denoted **M2** in merge code) with **`invn = 1/n`**: [`Dynamics.tke_field_from_velocity_moments!`](../utils/dynamics.jl), [`Dynamics.tke_field_from_sum_sq_dev_uvw!`](../utils/dynamics.jl). **Block Chan path:** scale merged **M2** / **C** with one **`invn`** per slab via `@. out = numer * invn` in `dataset_builder_impl.jl`.

## Scripts and experiments

- [`scripts/welford_test.jl`](../scripts/welford_test.jl) — pedagogy: Chan merge vs merging raw moments; Float32 cancellation demos.
- [`scripts/googleles_variance_numerics_one_timestep.jl`](../scripts/googleles_variance_numerics_one_timestep.jl) — real Zarr slice numerics.
- **Regenerate GoogleLES Arrow tables:** use `GoogleLES.build_tabular` from [`utils/build_training_data.jl`](../utils/build_training_data.jl) as in [`docs/googleles_build_tabular.md`](googleles_build_tabular.md) (project env + Zarr paths + output dir).

## References

- Chan, T. F., Golub, G. H., LeVeque, R. J. — updating formulas for sample mean and variance (parallel merge).
- Pebay, P. (2008) — parallel merge for covariance matrices (scalar 2×2 case is the pairwise cov merge used here). [https://www.osti.gov/servlets/purl/1028931]
