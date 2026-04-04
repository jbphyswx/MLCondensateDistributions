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

Valid (non-padded) sliding box averages; stride defaults to 1.
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

Geometric ladder from `nh_min = ceil(min_h/dx)` doubling until `min(nx,ny)`, plus `nh_min+1`
when it fits (doc example: `min_h/dx = 20` ⇒ include both 20 and 21 for truncated tiling).
"""
function truncated_horizontal_sizes(nx::Int, ny::Int, dx::T, min_h::T) where {T <: Real}
    nh_max = min(nx, ny)
    nh0 = max(1, ceil(Int, min_h / dx))
    nh0 > nh_max && return Int[nh_max]
    out = Int[]
    n = nh0
    while n <= nh_max
        push!(out, n)
        n == nh_max && break
        n = min(nh_max, n * 2)
    end
    if nh0 + 1 <= nh_max
        push!(out, nh0 + 1)
    end
    sort!(unique!(out))
    return out
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
