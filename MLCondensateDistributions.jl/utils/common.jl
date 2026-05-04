"""
Common utility functions used across the package.

This module contains generic utilities: data transformations, statistics helpers,
and I/O operations. These are backend-independent and may be used by visualization,
analysis, or training code.
"""
module Common

using Statistics: Statistics

export robust_limits, quantile_limits, is_constant, centers_to_edges, binned_median, write_metrics_table

"""
Compute robust plotting limits for a numeric vector.

The bounds are based on quantiles (default 0.5% and 99.5%), with fallback to
the finite min/max when the quantiles collapse or are not finite.

This is used to set axis limits in plots that contain outliers or extreme values.
"""
function robust_limits(vals::AbstractVector{<:Real}; qlo::Float64=0.005, qhi::Float64=0.995)
    finite_vals = Float64[x for x in vals if isfinite(x)]
    isempty(finite_vals) && return (0.0, 1.0)

    lo = Statistics.quantile(finite_vals, qlo)
    hi = Statistics.quantile(finite_vals, qhi)

    if !(isfinite(lo) && isfinite(hi)) || lo == hi
        lo = Statistics.minimum(finite_vals)
        hi = Statistics.maximum(finite_vals)
    end
    if lo == hi
        δ = abs(lo) > 0 ? 0.1 * abs(lo) : 1e-6
        return (lo - δ, hi + δ)
    end
    return (lo, hi)
end

"""
Axis-style bounds from empirical quantiles of finite `vals`, with optional outward margins.

Default `qlo`/`qhi` are **0.05** and **0.95** (typical probability-mass / diagnostic views).
After [`robust_limits`](@ref) (including its min/max and near-degenerate fallbacks), subtracts
`margin_lo` from the lower bound and adds `margin_hi` to the upper bound.

If `clamp_low` is set (e.g. `0.0` for nonnegative targets), the lower bound is `max(clamp_low, lo)`
after margins. If the interval is still invalid or collapsed, expands symmetrically around the midpoint
( same order of magnitude as [`robust_limits`](@ref) for `lo == hi`).
"""
function quantile_limits(
    vals::AbstractVector{<:Real};
    qlo::Float64=0.05,
    qhi::Float64=0.95,
    margin_lo::Float64=0.0,
    margin_hi::Float64=0.0,
    clamp_low::Union{Nothing,Float64}=nothing,
)
    lo, hi = robust_limits(vals; qlo=qlo, qhi=qhi)
    lo -= margin_lo
    hi += margin_hi
    if clamp_low !== nothing
        lo = max(Float64(clamp_low), lo)
    end
    if lo < hi
        return (lo, hi)
    end
    mid = (lo + hi) / 2
    δ = abs(mid) > 0 ? 0.1 * abs(mid) : 1e-6
    return (mid - δ, mid + δ)
end

"""
Return `true` when all finite elements in `vals` are equal.

This is used to special-case degenerate targets in distribution and scatter plots.
"""
function is_constant(vals::AbstractVector{<:Real})
    finite_vals = Float64[x for x in vals if isfinite(x)]
    isempty(finite_vals) && return true
    return Statistics.minimum(finite_vals) == Statistics.maximum(finite_vals)
end

"""
Convert a sequence of bin centers into plotting edges.

CairoMakie heatmaps require edge coordinates, while the analysis layer
naturally produces centers. This function computes uniform spacing extrapolation
to generate the full edge vector.
"""
function centers_to_edges(vals::AbstractVector{<:Real})
    n = length(vals)
    n == 0 && return Float64[]
    n == 1 && return Float64[Float64(vals[1]) - 0.5, Float64(vals[1]) + 0.5]

    edges = Vector{Float64}(undef, n + 1)
    edges[2:n] = (Float64.(vals[1:end-1]) .+ Float64.(vals[2:end])) ./ 2
    first_step = edges[2] - Float64(vals[1])
    last_step = Float64(vals[end]) - edges[n]
    edges[1] = Float64(vals[1]) - first_step
    edges[end] = Float64(vals[end]) + last_step
    return edges
end

"""
Compute median values of `y` in equal-width bins of `x`.

This is used for conditional diagnostic views (e.g., q-liquid vs qt) where
a coarse binning is more readable than a scatter plot.
"""
function binned_median(
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    nbins::Int=30,
    qlo::Float64=0.05,
    qhi::Float64=0.95,
    margin_lo::Float64=0.0,
    margin_hi::Float64=0.0,
)
    finite = isfinite.(x) .& isfinite.(y)
    xf = Float64.(x[finite])
    yf = Float64.(y[finite])
    isempty(xf) && return Float64[], Float64[]

    xlo, xhi = quantile_limits(xf; qlo=qlo, qhi=qhi, margin_lo=margin_lo, margin_hi=margin_hi)
    if xlo == xhi
        return Float64[xlo], Float64[Statistics.median(yf)]
    end

    edges = collect(range(xlo, xhi; length=nbins + 1))
    centers = Vector{Float64}(undef, nbins)
    meds = fill(NaN, nbins)

    for b in 1:nbins
        centers[b] = (edges[b] + edges[b + 1]) / 2
        in_bin = b == nbins ? (xf .>= edges[b]) .& (xf .<= edges[b + 1]) : (xf .>= edges[b]) .& (xf .< edges[b + 1])
        if any(in_bin)
            meds[b] = Statistics.median(yf[in_bin])
        end
    end

    keep = .!isnan.(meds)
    return centers[keep], meds[keep]
end

"""
Write a CSV metrics table from a model artifact metrics dictionary.

This is independent of the backend so downstream analysis can consume it
without worrying about whether plots were rendered.
"""
function write_metrics_table(metrics::Dict{Symbol, NamedTuple}, out_dir::String)
    rows = String[]
    push!(rows, "target,mse,rmse,mae,r2")
    for target in sort(collect(keys(metrics)); by=String)
        m = metrics[target]
        push!(rows, "$(target),$(m.mse),$(m.rmse),$(m.mae),$(m.r2)")
    end
    open(joinpath(out_dir, "metrics.csv"), "w") do io
        write(io, join(rows, "\n"))
    end
end

end # module
