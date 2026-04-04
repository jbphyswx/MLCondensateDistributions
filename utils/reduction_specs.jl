"""
Reduction specifications for multiscale dataset construction.

Dispatch targets: [`BlockReductionSpec`](@ref), [`SlidingConvolutionSpec`](@ref), [`HybridReductionSpec`](@ref).
See `docs/COARSENING_REDUCTION_IMPLEMENTATION_PLAN.md`.
"""
module ReductionSpecs

export AbstractReductionSpec,
    BlockReductionSpec,
    SlidingConvolutionSpec,
    HybridReductionSpec,
    truncated_block_extent,
    crop_ranges_3d,
    truncated_horizontal_sizes,
    block_reduction_triples,
    sliding_reduction_triples,
    subsample_closed_range,
    hybrid_sliding_extra_sizes_default,
    default_hybrid_sliding_windows,
    merge_reduction_metadata

"""Supertype for reduction strategies (block, sliding, hybrid)."""
abstract type AbstractReductionSpec end

"""
    BlockReductionSpec()

Non-overlapping 3D block means on a truncated subdomain (origin corner; remainder discarded).
Schedule is built from `min_h`, `max_dz`, and native grid — see [`truncated_horizontal_sizes`](@ref).
"""
struct BlockReductionSpec <: AbstractReductionSpec end

"""
    SlidingConvolutionSpec(; stride_h=1, stride_v=1, stride_z=1)

Valid (non-padded) sliding box averages. In `DatasetBuilderImpl`, strides default from
`spatial_info.sliding_outputs_*` (sparse placement) unless `sliding_stride_*` is set explicitly.
"""
Base.@kwdef struct SlidingConvolutionSpec <: AbstractReductionSpec
    stride_h::Int = 1
    stride_v::Int = 1
    stride_z::Int = 1
end

"""
    HybridReductionSpec(; sliding_spec=SlidingConvolutionSpec())

Runs truncated block reductions, then optional sliding passes from [`default_hybrid_sliding_windows`](@ref).
"""
Base.@kwdef struct HybridReductionSpec <: AbstractReductionSpec
    sliding_spec::SlidingConvolutionSpec = SlidingConvolutionSpec()
end

"""
    truncated_block_extent(n::Int, block::Int) -> (n_used, n_discarded)

`n_used = (n ÷ block) * block` cells from the low-index origin; remainder discarded.
"""
function truncated_block_extent(n::Int, block::Int)
    block >= 1 || throw(ArgumentError("block must be >= 1"))
    n_used = div(n, block) * block
    return (n_used, n - n_used)
end

"""
    crop_ranges_3d(nx, ny, nz, nxu, nyu, nzu)

Return ranges `1:nxu`, `1:nyu`, `1:nzu` after validation.
"""
function crop_ranges_3d(nx::Int, ny::Int, nz::Int, nxu::Int, nyu::Int, nzu::Int)
    (nxu <= nx && nyu <= ny && nzu <= nz) || throw(ArgumentError("crop exceeds domain"))
    return (1:nxu, 1:nyu, 1:nzu)
end

"""
    truncated_horizontal_sizes(nx, ny, dx, min_h) -> Vector{Int}

Horizontal block widths in **cells** (square `nh×nh`) for truncated block coarsening.

Schedule (see `docs/COARSENING_REDUCTION_IMPLEMENTATION_PLAN.md` §4.3):

- Start at `nh_min = ceil(min_h/dx)`.
- Close under multiplication by **2, 3, and 5** while `nh ≤ min(nx,ny)` (breadth-first on the factor monoid).
- Always include **`nh_max = min(nx,ny)`** if not already reached (full horizontal tile / domain scale).

This yields non-binary horizontal coverage (e.g. 20 → 40 → 60 → …) without the old `nh_min+1` extra rung.
"""
function truncated_horizontal_sizes(nx::Int, ny::Int, dx::T, min_h::T) where {T <: Real}
    nh_max = min(nx, ny)
    nh_min = max(1, ceil(Int, min_h / dx))
    nh_min > nh_max && return Int[nh_max]
    reached = Set{Int}()
    queue = Int[nh_min]
    push!(reached, nh_min)
    i = 1
    while i <= length(queue)
        s = queue[i]
        i += 1
        for p in (2, 3, 5)
            sn = s * p
            if sn <= nh_max && !(sn in reached)
                push!(reached, sn)
                push!(queue, sn)
            end
        end
    end
    if nh_max ∉ reached
        push!(reached, nh_max)
    end
    return sort!(collect(reached))
end

"""
    block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz; square_horizontal=true)

All `(nh, nh, fz)` (or `(nh, mh, fz)` if `square_horizontal` false — not implemented) where
`nh` comes from [`truncated_horizontal_sizes`](@ref) and `fz` divides `nz` with `fz * dz_ref <= max_dz`.
"""
function block_reduction_triples(
    nx::Int,
    ny::Int,
    nz::Int,
    dx::T,
    min_h::T,
    dz_ref::T,
    max_dz::T;
    square_horizontal::Bool = true,
)::Vector{NTuple{3,Int}} where {T <: Real}
    nhs = truncated_horizontal_sizes(nx, ny, dx, min_h)
    triples = NTuple{3,Int}[]
    for nh in nhs
        if square_horizontal
            for fz in 1:nz
                mod(nz, fz) != 0 && continue
                fz * dz_ref > max_dz && continue
                push!(triples, (nh, nh, fz))
            end
        end
    end
    unique!(triples)
    sort!(triples; by = t -> (t[3], t[1], t[2]))
    return triples
end

"""
    subsample_closed_range(lo, hi, n) -> Vector{Int}

Up to `n` distinct integers in `[lo, hi]`, approximately evenly spaced (endpoints included
when `n >= 2`). If `hi - lo + 1 <= n`, returns `lo:hi`.
"""
function subsample_closed_range(lo::Int, hi::Int, n::Int)::Vector{Int}
    n < 1 && return Int[]
    lo > hi && return Int[]
    len = hi - lo + 1
    len <= n && return collect(lo:hi)
    n == 1 && return Int[clamp(lo + div(hi - lo, 2), lo, hi)]
    out = Int[]
    for i in 1:n
        v = lo + round(Int, (i - 1) * (hi - lo) / (n - 1))
        push!(out, clamp(v, lo, hi))
    end
    sort!(unique!(out))
    return out
end

"""
    sliding_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz; horizontal_budget=5)

Like [`block_reduction_triples`](@ref) but horizontal window sizes are **subsampled** to at
most `horizontal_budget` values evenly spaced between `ceil(min_h/dx)` and `min(nx,ny)`,
instead of the full geometric ladder (keeps sliding-only mode cheap on large domains).
"""
function sliding_reduction_triples(
    nx::Int,
    ny::Int,
    nz::Int,
    dx::T,
    min_h::T,
    dz_ref::T,
    max_dz::T;
    horizontal_budget::Int = 5,
    square_horizontal::Bool = true,
)::Vector{NTuple{3,Int}} where {T <: Real}
    horizontal_budget >= 1 || throw(ArgumentError("horizontal_budget must be >= 1"))
    nh_min = max(1, ceil(Int, min_h / dx))
    nh_max = min(nx, ny)
    whs = subsample_closed_range(nh_min, nh_max, horizontal_budget)
    triples = NTuple{3,Int}[]
    for wh in whs
        if square_horizontal
            for fz in 1:nz
                mod(nz, fz) != 0 && continue
                fz * dz_ref > max_dz && continue
                push!(triples, (wh, wh, fz))
            end
        end
    end
    unique!(triples)
    sort!(triples; by = t -> (t[3], t[1], t[2]))
    return triples
end

"""
    hybrid_sliding_extra_sizes_default(nx, ny, dx, min_h, block_nhs; budget=5)

Horizontal window sizes for hybrid **sliding** passes: up to `budget` subsampled sizes in
`[nh_min, nh_max]` that are **not** already covered by block reductions (`block_nhs`). If
that set is empty, falls back to [`default_hybrid_sliding_windows`](@ref).
"""
function hybrid_sliding_extra_sizes_default(
    nx::Int,
    ny::Int,
    dx::T,
    min_h::T,
    block_nhs::AbstractSet{Int};
    budget::Int = 5,
)::Vector{Int} where {T <: Real}
    budget >= 1 || throw(ArgumentError("budget must be >= 1"))
    nh_min = max(1, ceil(Int, min_h / dx))
    nh_max = min(nx, ny)
    cand = subsample_closed_range(nh_min, nh_max, budget)
    extra = filter(w -> !(w in block_nhs), cand)
    if !isempty(extra)
        return extra
    end
    return default_hybrid_sliding_windows(nx, ny, dx, min_h)
end

"""
    default_hybrid_sliding_windows(nx, ny, dx, min_h)

Extra sliding window sizes (horizontal edge length in cells) between coarse block scales.
Returns at most one midpoint between `nh_min` and `min(nx,ny)` when it differs from block ladder.
"""
function default_hybrid_sliding_windows(nx::Int, ny::Int, dx::T, min_h::T) where {T <: Real}
    nh_min = max(1, ceil(Int, min_h / dx))
    nh_max = min(nx, ny)
    nh_mid = div(nh_min + nh_max, 2)
    if nh_mid > nh_min && nh_mid < nh_max && nh_mid != nh_min * 2
        return Int[nh_mid]
    end
    return Int[]
end

function merge_reduction_metadata(metadata, pairs::Vararg{Pair{Symbol}, N}) where {N}
    if metadata isa AbstractDict{Symbol}
        d = Dict{Symbol,Any}(metadata)
        for (k, v) in pairs
            d[k] = v
        end
        return d
    end
    return (; metadata..., pairs...)
end

end # module ReductionSpecs
