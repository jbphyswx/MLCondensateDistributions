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
    effective_fz_max,
    vertical_block_factors_from_tower,
    block_reduction_triples,
    sliding_reduction_triples,
    subsample_closed_range,
    hybrid_sliding_extra_sizes_default,
    hybrid_sliding_extra_vertical_default,
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
    effective_fz_max(nz, dz_ref, max_dz) -> Int

Largest integer `fz` in `1:nz` with `fz * dz_ref ≤ max_dz` (same rule as the legacy divisor loop).
Returns `0` if no such `fz` exists.
"""
function effective_fz_max(nz::Int, dz_ref::T, max_dz::T) where {T <: Real}
    nz < 1 && return 0
    f = 0
    @inbounds for fz in 1:nz
        (fz * dz_ref <= max_dz) || break
        f = fz
    end
    return f
end

function _primes_asc_up_to(n::Int)::Vector{Int}
    n < 2 && return Int[]
    is_comp = falses(n)
    sqrtn = isqrt(n)
    @inbounds for p in 2:sqrtn
        is_comp[p] && continue
        q = p * p
        while q <= n
            is_comp[q] = true
            q += p
        end
    end
    out = Int[]
    @inbounds for p in 2:n
        is_comp[p] || push!(out, p)
    end
    return out
end

function _vertical_tower_dfs!(out::Set{Int}, fz_eff::Int, nzc::Int, fz_max::Int)
    nzc < 2 && return nothing
    for p in _primes_asc_up_to(nzc)
        div(nzc, p) < 1 && continue
        fz_child = fz_eff * p
        fz_child > fz_max && continue
        push!(out, fz_child)
        nzc_child = div(nzc, p)
        _vertical_tower_dfs!(out, fz_child, nzc_child, fz_max)
    end
    return nothing
end

"""
    vertical_block_factors_from_tower(nz, dz_ref, max_dz; vertical_budget=5)

Vertical **block** factors `fz` (native layers per coarse layer), mirroring the horizontal tower:

- `fz_max = effective_fz_max(nz, dz_ref, max_dz)`.
- **Seeds** in `1 : ⌊fz_max/2⌋` (or `1:fz_max` if that range is empty), subsampled with `vertical_budget`
  (or full range if `vertical_budget === nothing`).
- **Tower:** from each seed, prime-sized **1D** pooling steps along `z` on the truncated coarse grid
  (`⌊nz/fz⌋` coarse levels after grouping by `fz`), while `fz_eff * p ≤ fz_max`.

Sorted distinct integers. Empty if `fz_max == 0`.
"""
function vertical_block_factors_from_tower(
    nz::Int,
    dz_ref::T,
    max_dz::T;
    vertical_budget::Union{Nothing,Int} = 5,
)::Vector{Int} where {T <: Real}
    fz_max = effective_fz_max(nz, dz_ref, max_dz)
    fz_max < 1 && return Int[]
    out = Set{Int}()
    fz_half = fld(fz_max, 2)
    seeds = if fz_half >= 1
        vertical_budget === nothing ? collect(1:fz_half) :
        subsample_closed_range(1, fz_half, vertical_budget)
    else
        vertical_budget === nothing ? collect(1:fz_max) :
        subsample_closed_range(1, fz_max, vertical_budget)
    end
    isempty(seeds) && push!(seeds, 1)
    for fz0 in seeds
        fz0 < 1 && continue
        fz0 > fz_max && continue
        nzu = div(nz, fz0) * fz0
        nzu < fz0 && continue
        nzc = div(nzu, fz0)
        push!(out, fz0)
        _vertical_tower_dfs!(out, fz0, nzc, fz_max)
    end
    return sort!(collect(out))
end

"""
    truncated_horizontal_sizes(nx, ny, dx, min_h; horizontal_budget=5) -> Vector{Int}

Horizontal block widths in **cells** (square `nh×nh`) for truncated block coarsening.

- `nh_min = ceil(min_h/dx)`, `nh_max = min(nx, ny)`.
- **Default (`horizontal_budget` an `Int ≥ 1`):** up to that many values, evenly subsampled in
  `[nh_min, nh_max]` via [`subsample_closed_range`](@ref) — same cost idea as sliding window budgets
  (avoids very large default work when `N_h = nh_max - nh_min + 1` is big).
- **`horizontal_budget === nothing`:** every integer `nh_min:nh_max` (research / small domains).

To request the full ladder without `nothing`, set `horizontal_budget` ≥ `nh_max - nh_min + 1`
(`subsample_closed_range` then returns the whole range).

**Tower / cost:** `_block_truncated_horizontal_cache!` reuses the largest cached `s | nh` and pools
by `nh÷s` when possible.
"""
function truncated_horizontal_sizes(
    nx::Int,
    ny::Int,
    dx::T,
    min_h::T;
    horizontal_budget::Union{Nothing,Int} = 5,
) where {T <: Real}
    nh_max = min(nx, ny)
    nh_min = max(1, ceil(Int, min_h / dx))
    nh_min > nh_max && return Int[nh_max]
    horizontal_budget === nothing && return collect(nh_min:nh_max)
    horizontal_budget < 1 && throw(ArgumentError("horizontal_budget must be ≥ 1 or nothing"))
    return subsample_closed_range(nh_min, nh_max, horizontal_budget)
end

"""
    block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz; square_horizontal=true, horizontal_budget=5)

All `(nh, nh, fz)` where `nh` comes from [`truncated_horizontal_sizes`](@ref) and each `fz` comes from
[`vertical_block_factors_from_tower`](@ref) with `vertical_budget = horizontal_budget`.
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
    horizontal_budget::Union{Nothing,Int} = 5,
)::Vector{NTuple{3,Int}} where {T <: Real}
    nhs = truncated_horizontal_sizes(nx, ny, dx, min_h; horizontal_budget)
    fzs = vertical_block_factors_from_tower(nz, dz_ref, max_dz; vertical_budget = horizontal_budget)
    triples = NTuple{3,Int}[]
    for nh in nhs
        if square_horizontal
            for fz in fzs
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

Horizontal window sizes are **subsampled** to at most `horizontal_budget` values in
`[ceil(min_h/dx), min(nx,ny)]`. Vertical window heights `fz` are **subsampled** to at most
`horizontal_budget` values in `[1, effective_fz_max(nz, dz_ref, max_dz)]` (no divisor requirement;
valid-box coarsening truncates like block means).
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
    fz_max = effective_fz_max(nz, dz_ref, max_dz)
    fzs = fz_max >= 1 ? subsample_closed_range(1, fz_max, horizontal_budget) : Int[]
    triples = NTuple{3,Int}[]
    for wh in whs
        if square_horizontal
            for fz in fzs
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
that set is empty and the block schedule already covers **every** `nh` in `nh_min:nh_max`,
returns **no** extras (dense block ladder — sliding gap-fill not needed). Otherwise falls back
to [`default_hybrid_sliding_windows`](@ref) (sparse block schedules only).
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
    if nh_min <= nh_max && all(n -> n in block_nhs, nh_min:nh_max)
        return Int[]
    end
    fb = default_hybrid_sliding_windows(nx, ny, dx, min_h)
    return filter(w -> !(w in block_nhs), fb)
end

"""
    hybrid_sliding_extra_vertical_default(nz, dz_ref, max_dz, block_fzs; budget=5)

Vertical window heights for hybrid **sliding** passes: up to `budget` subsampled integers in
`[⌊fz_max/2⌋ + 1, fz_max]` (with `fz_max = effective_fz_max(...)`) that are **not** in `block_fzs`.
If that band is empty, returns `Int[]`. If every `fz` in the band is already a block factor, returns
`Int[]` (same “dense ladder” idea as horizontal extras).
"""
function hybrid_sliding_extra_vertical_default(
    nz::Int,
    dz_ref::T,
    max_dz::T,
    block_fzs::AbstractSet{Int};
    budget::Int = 5,
)::Vector{Int} where {T <: Real}
    budget >= 1 || throw(ArgumentError("budget must be >= 1"))
    fz_max = effective_fz_max(nz, dz_ref, max_dz)
    fz_max < 1 && return Int[]
    lo = fld(fz_max, 2) + 1
    hi = fz_max
    lo > hi && return Int[]
    cand = subsample_closed_range(lo, hi, budget)
    extra = filter(w -> !(w in block_fzs), cand)
    if !isempty(extra)
        return extra
    end
    if all(z -> z in block_fzs, lo:hi)
        return Int[]
    end
    return filter(w -> !(w in block_fzs), cand)
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
