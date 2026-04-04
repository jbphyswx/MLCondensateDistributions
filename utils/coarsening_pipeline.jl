module CoarseningPipeline

include("array_utils.jl")
using .ArrayUtils

export build_horizontal_levels,
    coarsen_fields_at_level,
    coarsen_fields_vertical_at_level,
    coarsen_products_vertical_at_level,
    coarsen_dz_profile_2x_at_level,
    coarsen3d_vertical_mean,
    coarsen_products_at_level,
    covariance_from_moments,
    build_horizontal_multilevel_views

"""
    build_horizontal_levels(nx, ny, dx_native; seeds=(1,), min_h=0, include_full_domain=true)

Build horizontal coarsening levels for possibly non-square domains.
Generates factors from arbitrary odd seed tuples via seed_factor_ladder.
Can optionally include a guaranteed full-domain level.

Returns vector of named tuples with fx, fy, resolution_h.
"""
function build_horizontal_levels(
    nx::Int,
    ny::Int,
    dx_native::T;
    seeds::NTuple{N, Int}=(1,),
    min_h::T=zero(T),
    include_full_domain::Bool=true,
) where {T <: Real, N}
    nx >= 1 || throw(ArgumentError("nx must be >= 1"))
    ny >= 1 || throw(ArgumentError("ny must be >= 1"))
    dx_native > zero(T) || throw(ArgumentError("dx_native must be > 0"))

    limit = min(nx, ny)
    base = build_schedule_from_seeds(limit, dx_native; seeds=seeds, min_h=min_h, include_full_domain=false)

    # Estimate output size: base levels + possibly full domain
    est_size = length(base) + (include_full_domain ? 1 : 0)
    levels = NamedTuple{(:fx, :fy, :resolution_h), Tuple{Int, Int, T}}[]
    sizehint!(levels, est_size)
    
    for row in base
        f = row.factor
        # Keep square pooling for now to match existing behavior.
        push!(levels, (fx=f, fy=f, resolution_h=T(f) * dx_native))
    end

    if include_full_domain
        # Always append exact full-domain aggregate if not already represented.
        if isempty(levels) || !(last(levels).fx == nx && last(levels).fy == ny)
            push!(levels, (fx=nx, fy=ny, resolution_h=T(nx) * dx_native))
        end
    end

    return levels
end

"""
    coarsen_fields_at_level(fields, fx, fy)

Coarsen all 3D fields at one horizontal level.
"""
function coarsen_fields_at_level(
    fields::AbstractDict{K, A},
    fx::Int,
    fy::Int,
) where {K, T <: Real, A <: AbstractArray{T, 3}}
    return coarsen_fields_horizontal(fields, fx, fy)
end

function coarsen_fields_at_level(
    fields::NamedTuple{N, V},
    fx::Int,
    fy::Int,
) where {N, V <: Tuple}
    vals = map(arr -> coarsen3d_horizontal_mean(arr, fx, fy), values(fields))
    return NamedTuple{N}(vals)
end

"""
    coarsen_fields_vertical_at_level(fields, fz)

Coarsen all 3D fields at one vertical level.
"""
function coarsen_fields_vertical_at_level(
    fields::AbstractDict{K, A},
    fz::Int,
) where {K, T <: Real, A <: AbstractArray{T, 3}}
    return coarsen_fields_vertical(fields, fz)
end

function coarsen_fields_vertical_at_level(
    fields::NamedTuple{N, V},
    fz::Int,
) where {N, V <: Tuple}
    vals = map(arr -> coarsen3d_vertical_mean(arr, fz), values(fields))
    return NamedTuple{N}(vals)
end

"""
    coarsen_products_vertical_at_level(products, fz)

Coarsen already-computed `<x*y>` product fields vertically by factor `fz`.
"""
function coarsen_products_vertical_at_level(
    products::AbstractDict{K, A},
    fz::Int,
) where {K, T <: Real, A <: AbstractArray{T, 3}}
    return coarsen_fields_vertical(products, fz)
end

function coarsen_products_vertical_at_level(
    products::NamedTuple{N, V},
    fz::Int,
) where {N, V <: Tuple}
    vals = map(arr -> coarsen3d_vertical_mean(arr, fz), values(products))
    return NamedTuple{N}(vals)
end

"""
    coarsen_dz_profile_2x_at_level(dz_profile)

Coarsen vertical thickness profile by summing adjacent pairs.
"""
function coarsen_dz_profile_2x_at_level(dz_profile::AbstractVector{<:Real})
    return coarsen_dz_profile_2x(dz_profile)
end

"""
    coarsen_products_at_level(fields, product_pairs, fx, fy)

Compute coarse `<x*y>` fields at one horizontal level from fine fields.
`product_pairs` maps output-name => (x_name, y_name).
Coarsens the product directly without intermediate allocation.
"""
function coarsen_products_at_level(
    fields::AbstractDict{K, A},
    product_pairs::AbstractDict{K, Tuple{K, K}},
    fx::Int,
    fy::Int,
) where {K, T <: Real, A <: AbstractArray{T, 3}}
    out = Dict{K, Array{T, 3}}()
    for (out_name, (x_name, y_name)) in pairs(product_pairs)
        x = fields[x_name]
        y = fields[y_name]
        size(x) == size(y) || throw(DimensionMismatch("Fields for product $(out_name) have mismatched sizes"))
        
        # Coarsen product directly without intermediate allocation
        nx, ny, nz = size(x)
        nxo = div(nx, fx)
        nyo = div(ny, fy)
        prod_coarse = similar(x, nxo, nyo, nz)
        
        inv_area = one(T) / T(fx * fy)
        @inbounds for k in 1:nz
            for j in 1:nyo
                base_j = (j - 1) * fy + 1
                for i in 1:nxo
                    base_i = (i - 1) * fx + 1
                    acc = zero(T)
                    for jj in 0:(fy - 1)
                        for ii in 0:(fx - 1)
                            acc += x[base_i + ii, base_j + jj, k] * y[base_i + ii, base_j + jj, k]
                        end
                    end
                    prod_coarse[i, j, k] = acc * inv_area
                end
            end
        end
        
        out[out_name] = prod_coarse
    end
    return out
end

@inline _field_name_key(name::Symbol) = name
@inline _field_name_key(name::AbstractString) = Symbol(name)

@inline _container_get_field(fields::AbstractDict, name) = fields[name]
@inline _container_get_field(fields::NamedTuple, name::Symbol) = getproperty(fields, name)
@inline _container_get_field(fields::NamedTuple, name::AbstractString) = getproperty(fields, Symbol(name))

function coarsen_products_at_level(
    fields::NamedTuple{FN, FV},
    product_pairs::NamedTuple{PN, PV},
    fx::Int,
    fy::Int,
) where {FN, FV <: Tuple, PN, PV <: Tuple}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        x = _container_get_field(fields, _field_name_key(x_name))
        y = _container_get_field(fields, _field_name_key(y_name))
        size(x) == size(y) || throw(DimensionMismatch("Fields for product $(out_name) have mismatched sizes"))

        nx, ny, nz = size(x)
        nxo = div(nx, fx)
        nyo = div(ny, fy)
        prod_coarse = similar(x, nxo, nyo, nz)

        T = eltype(x)
        inv_area = one(T) / T(fx * fy)
        @inbounds for k in 1:nz
            for j in 1:nyo
                base_j = (j - 1) * fy + 1
                for i in 1:nxo
                    base_i = (i - 1) * fx + 1
                    acc = zero(T)
                    for jj in 0:(fy - 1)
                        for ii in 0:(fx - 1)
                            acc += x[base_i + ii, base_j + jj, k] * y[base_i + ii, base_j + jj, k]
                        end
                    end
                    prod_coarse[i, j, k] = acc * inv_area
                end
            end
        end
        prod_coarse
    end
    return NamedTuple{PN}(vals)
end

"""
    covariance_from_moments(mean_xy, mean_x, mean_y)

Compute covariance field as `<xy> - <x><y>`.
"""
function covariance_from_moments(
    mean_xy::AbstractArray{T, 3},
    mean_x::AbstractArray{T, 3},
    mean_y::AbstractArray{T, 3},
) where {T <: Real}
    size(mean_xy) == size(mean_x) == size(mean_y) || throw(DimensionMismatch("moment arrays must share shape"))
    return mean_xy .- (mean_x .* mean_y)
end

"""
    build_horizontal_multilevel_views(fields, dx_native; seeds=(1,), min_h=0, include_full_domain=true, product_pairs=Dict())

Builds a vector of per-level bundles suitable for a staged replacement of
`process_abstract_chunk` horizontal logic.

User supplies arbitrary odd seed tuple; schedule is generated via seed_factor_ladder.

Each element contains:
- `fx`, `fy`
- `resolution_h`
- `means` (coarsened fields)
- `products` (coarsened `<x*y>` fields)
"""
function build_horizontal_multilevel_views(
    fields::AbstractDict{K, A},
    dx_native::T;
    seeds::NTuple{N, Int}=(1,),
    min_h::T=zero(T),
    include_full_domain::Bool=true,
    product_pairs::AbstractDict{K, Tuple{K, K}}=Dict{K, Tuple{K, K}}(),
) where {K, T <: Real, A <: AbstractArray{T, 3}, N}
    isempty(fields) && return NamedTuple[]

    first_key = first(keys(fields))
    nx, ny, _ = size(fields[first_key])
    levels = build_horizontal_levels(nx, ny, dx_native; seeds=seeds, min_h=min_h, include_full_domain=include_full_domain)

    # Incremental cache keyed by absolute factor.
    # For each target factor f, reuse the largest previously computed divisor s|f
    # and coarsen by ratio r=f/s, instead of always recomputing from fine fields.
    means_cache = Dict{Int, Dict{K, Array{T, 3}}}()
    prods_cache = Dict{Int, Dict{K, Array{T, 3}}}()
    done_factors = Int[]
    sizehint!(done_factors, length(levels))

    out = Vector{NamedTuple}(undef, length(levels))
    for (idx, lvl) in enumerate(levels)
        f = lvl.fx

        best_src = 0
        for s in done_factors
            if (f % s == 0) && (s > best_src)
                best_src = s
            end
        end

        means = if best_src == 0
            coarsen_fields_at_level(fields, f, f)
        else
            r = div(f, best_src)
            coarsen_fields_at_level(means_cache[best_src], r, r)
        end

        prods = if isempty(product_pairs)
            Dict{K, Array{T, 3}}()
        elseif best_src == 0
            coarsen_products_at_level(fields, product_pairs, f, f)
        else
            r = div(f, best_src)
            coarsen_fields_at_level(prods_cache[best_src], r, r)
        end

        means_cache[f] = means
        if !isempty(product_pairs)
            prods_cache[f] = prods
        end
        push!(done_factors, f)

        out[idx] = (
            fx=lvl.fx,
            fy=lvl.fy,
            resolution_h=lvl.resolution_h,
            means=means,
            products=prods,
        )
    end

    return out
end

end # module
