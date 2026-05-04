# =========================================================
# Stable Parallel Variance Computation in Julia
# =========================================================
#
# **Main MLCD-relevant point:** combining coarse summaries.
# - If each child block gives **(n, μ, M2)** with `M2 = Σ(x−μ)²`, Chan/Welford **merge**
#   (`merge_variance_welford`) is stable and works **without raw samples**—this is the “Welford
#   above the first level” idea.
# - If you only store **⟨x⟩** and **⟨x²⟩** per block and later **average those fields** (then do
#   `⟨x²⟩−⟨x⟩²`), that matches exact nested means in theory but loses information and is numerically
#   fragile in Float32 when |x| is large and variance is tiny (`merge_variance_from_moments`).
#   `run_welford_test()` contrasts merged **M2** vs merged **moments** on two chunks.
#
# **Per-chunk** `(n, μ, M2)` can be obtained in several equivalent-in-principle ways:
# - `chunk_stats_welford`: compute `μ`, then `M2 = Σ(x−μ)²` (two passes over the chunk).
# - `chunk_stats_online_welford`: classic one-pass Welford (different rounding than two-pass in
#   general, but the same **kind** of summary for merging).
# For a fixed Float32 vector, both feed the **same** merge function; neither is “the merge”—the
# merge is `merge_variance_welford`. If your pain is **only** merge quality, either per-chunk M2
# recipe is usually fine; the big structural change is **storing and merging M2** (and analogous
# quantities for covariances), not swapping `Statistics.var` in isolation.
#
# **`Statistics.var` (side note):** on one vector it often avoids the worst `⟨x²⟩−⟨x⟩²` cancellation
# because it does **not** use that formula internally. MLCD still can’t “just call var” everywhere:
# fused 3D reductions, many fields, **⟨xy⟩** covariances, and coarsening that averages **moment
# fields**—so the actionable parallel to this script is **Welford-style merge of sufficient stats**,
# not `var` per block alone.
#
# **`conv3d_block_mean`:** that kernel only accumulates **one** sum per voxel (then `* inv_vol`).
# Second moments in the builder are separate fields; the negative-variance demo below uses the same
# **algebra** as `mean_sq - mean*mean` on Float32 raw moments (`chunk_stats_moments`), not a special
# different loop than `sum(x)/n` and `sum(x^2)/n` (for a `Vector{Float32}` they match in practice).

using Statistics: Statistics
using Random: Random

# ---------------------------------------------------------
# Compute statistics for a single data chunk
# ---------------------------------------------------------
"""
    chunk_stats_welford(x)

Per-chunk sufficient stats for **parallel merge**: first `μ = sum(x)/n`, then `M2 = Σ(x-μ)²`
(two passes). Feeds `merge_variance_welford`; this is **not** “online Welford,” just a convenient
way to get **M2** for the merge demo.
"""
function chunk_stats_welford(x::AbstractVector)
    n = length(x)
    μ = sum(x) / n
    M2 = sum((xi - μ)^2 for xi in x)
    return n, μ, M2
end

"""
    chunk_stats_online_welford(x)

Same **outputs** as `chunk_stats_welford` for merge purposes: `(n, μ, M2)` with population variance
`M2/n`. Implementation is classic **one-pass** Welford (rounding can differ slightly from the
two-pass `Σ(x−μ)²` formula). Either is interchangeable as input to `merge_variance_welford`.
"""
function chunk_stats_online_welford(x::AbstractVector{FT}) where {FT <: AbstractFloat}
    nloc = zero(FT)
    meanv = zero(FT)
    M2 = zero(FT)
    @inbounds for i in eachindex(x)
        xv = x[i]
        nloc += one(FT)
        delta = xv - meanv
        meanv += delta / nloc
        delta2 = xv - meanv
        M2 += delta * delta2
    end
    ni = Int(nloc)
    return ni, meanv, M2
end

"""
    chunk_stats_moments(x)

Compute summary statistics for a data chunk `x` using the raw moments method.

Returns:
- n       :: Int    → number of samples
- μ       :: Real   → mean of the chunk
- mean_sq :: Real   → mean of the squared values (Second Raw Moment)
"""
function chunk_stats_moments(x::AbstractVector)
    n = length(x)
    μ = sum(x) / n
    mean_sq = sum(xi^2 for xi in x) / n
    return n, μ, mean_sq
end

"""Population variance from raw moments: `mean_sq - μ*μ` (same as `StatisticalMethods.covariance_from_moments!` on diagonals)."""
function variance_from_raw_moments(μ, mean_sq)
    return mean_sq - μ * μ
end

# ---------------------------------------------------------
# Merge statistics from two independent chunks
# ---------------------------------------------------------
"""
    merge_variance_welford(n1, μ1, M2_1, n2, μ2, M2_2)

Combine two independent chunk statistics into one using Welford's algorithm.

Arguments:
- n1, μ1, M2_1 → statistics for chunk A
- n2, μ2, M2_2 → statistics for chunk B

Returns:
- n   → total number of samples (n1 + n2)
- μ   → combined mean
- M2  → combined sum of squared deviations
- var → population variance = M2 / n
"""
function merge_variance_welford(n1, μ1, M2_1, n2, μ2, M2_2)
    n = n1 + n2
    δ = μ2 - μ1                # difference in chunk means
    μ = (n1 * μ1 + n2 * μ2) / n  # weighted mean
    M2 = M2_1 + M2_2 + δ^2 * (n1 * n2 / n)  # combined sum of squared deviations
    var = M2 / n               # population variance
    return n, μ, M2, var
end

function why_not_do_this_if_N1_eq_N2(μ1, var1, μ2, var2) # var1 = M2_1/n, var2 = M2_2/n
    μ = (μ1 + μ2) / 2
    var = (var1 + var2) * (1/2) + (μ1 - μ2)^2 / 4
    return μ, var
end

"""
    merge_variance_from_moments(n1, μ1, mean_sq1, n2, μ2, mean_sq2)

Combine two independent chunk statistics into one using raw moments.

Returns:
- n       → total number of samples (n1 + n2)
- μ       → combined mean
- mean_sq → combined mean of squared values
- var     → population variance = mean_sq - μ^2
"""
function merge_variance_from_moments(n1, μ1, mean_sq1, n2, μ2, mean_sq2)
    n = n1 + n2
    μ = (n1 * μ1 + n2 * μ2) / n
    # We must weight the means by their sample sizes to combine them correctly
    mean_sq = (n1 * mean_sq1 + n2 * mean_sq2) / n
    var = mean_sq - μ^2
    return n, μ, mean_sq, var
end

# ---------------------------------------------------------
# Testing & Example usage
# ---------------------------------------------------------
function run_welford_test(;mean::Float32 = Float32(300), scale::Float32 = Float32(0.002), n::Int = 64*64*64)::Nothing
    # Change to Float64 to see the catastrophic cancellation error disappear
    FT = Float32 

    Random.seed!(7)



    # # Example: simulate two chunks
    chunkA = randn(FT, n ÷ 2) .* scale .+ mean 
    chunkB = randn(FT, n ÷ 2) .* scale .+ mean

    # 1. Welford's Approach
    n1_w, μ1_w, M2_1 = chunk_stats_welford(chunkA)
    n2_w, μ2_w, M2_2 = chunk_stats_welford(chunkB)
    
    n_total_w, μ_total_w, M2_total, var_total_w = merge_variance_welford(n1_w, μ1_w, M2_1, n2_w, μ2_w, M2_2)

    # 2. Raw Moments Approach
    n1_m, μ1_m, mean_sq1 = chunk_stats_moments(chunkA)
    n2_m, μ2_m, mean_sq2 = chunk_stats_moments(chunkB)
    
    n_total_m, μ_total_m, mean_sq_total, var_total_m = merge_variance_from_moments(n1_m, μ1_m, mean_sq1, n2_m, μ2_m, mean_sq2)

    # truth
    true_mean = (sum(chunkA) + sum(chunkB)) / (n1_w + n2_w)
    true_var = (sum((chunkA .- true_mean).^2) + sum((chunkB .- true_mean).^2)) / (n1_w + n2_w)
    true_mean_from_f64 = FT(sum(Float64.(chunkA)) + sum(Float64.(chunkB))) / (n1_w + n2_w)
    true_var_from_f64 = (FT(sum((Float64.(chunkA) .- true_mean_from_f64).^2)) + FT(sum((Float64.(chunkB) .- true_mean_from_f64).^2))) / (n1_w + n2_w)


    # Print comparisons
    println("--- Results ---")
    println("Total samples = $n_total_m")
    println("Combined mean | Truth: $true_mean | Truth from f64: $true_mean_from_f64 | Welford: $μ_total_w | Moments: $μ_total_m")
    println("Welford sum of sq deviations (M2) : ", M2_total)
    println("Moments mean of squares (mean_sq) : ", mean_sq_total)
    println("Combined variance | Truth: $true_var | Truth from f64: $true_var_from_f64 | Welford: $var_total_w | Moments: $var_total_m")
    println("Variance difference (Welford - Moments): ", var_total_w - var_total_m)
end

"""
    run_pipeline_style_negative_demo()

Uses **small** fluctuations around 300 (`θ_li`-like) so Float32 **`mean_sq - μ²`** from
`chunk_stats_moments` goes **negative**, while `Statistics.var(..., corrected=false)` and
`M2/n` from Welford stay sensible. Same phenomenon as `googleles_variance_numerics_one_timestep.jl`
on a flat vector (here `chunk_stats_moments` matches an explicit `invn` loop bit-for-bit).
"""
function run_pipeline_style_negative_demo(;mean::Float32 = Float32(300), scale::Float32 = Float32(0.002), n::Int = 64*64*64)::Nothing
    Random.seed!(7)
    x = randn(Float32, n) .* scale .+ mean
    μ_true, var_true = Statistics.mean(x), Statistics.var(x; corrected=false)
    n, μ_p, mean_sq_p = chunk_stats_moments(x)
    _, μ_p_f64, mean_sq_p_f64 = chunk_stats_moments(Float64.(x))
    var_moments = variance_from_raw_moments(μ_p, mean_sq_p)
    var_moments_f64 = variance_from_raw_moments(μ_p_f64, mean_sq_p_f64)
    _, _, M2_on = chunk_stats_online_welford(x)
    var_online = M2_on / Float32(n)
    s1 = sum(Float64, x)
    s2 = sum(z -> Float64(z)^2, x)
    var_ref = Float32(s2 / n - (s1 / n)^2)

    println()
    println("=== Float32 raw-moment subtraction demo (seed 7, $(length(x)) samples) === :: true mean: $μ_true | true var: $var_true")
    println("  Population var (Float64 reference on same Float32 values): ", var_ref)
    println("  ⟨x²⟩−⟨x⟩² from chunk_stats_moments (Float32):              var_moments: $var_moments | var_moments_f64: $var_moments_f64")
    println("  Online Welford Float32 (M2/n):                            ", var_online)
    println()
    println("  If ⟨x²⟩−⟨x⟩² < 0: catastrophic cancellation in the **moment subtraction**.")
    println("  Statistics.var avoids that here; online Welford does too, without Float64 arrays.")
    println("=== end ===")
    println()
    return nothing
end

function run_all_welford_scripts(;mean::Float32 = Float32(300), scale::Float32 = Float32(0.002), n::Int = 64*64*64)::Nothing
    run_welford_test(mean=mean, scale=scale, n=n)
    run_pipeline_style_negative_demo(;mean=mean, scale=scale, n=n)
end
    
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_welford_scripts()
end