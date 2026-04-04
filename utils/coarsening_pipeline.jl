module CoarseningPipeline

using Statistics: Statistics

include("array_utils.jl")
using .ArrayUtils

@inline uniform_stride_for_valid_box(n::Int, w::Int, k::Int) = ArrayUtils.uniform_stride_for_valid_box(n, w, k)
@inline valid_box_anchor_starts(n::Int, w::Int, k::Int) = ArrayUtils.valid_box_anchor_starts(n, w, k)

export build_horizontal_levels,
    build_convolutional_coarsening_triples,
    coarsen_fields_at_level,
    coarsen_fields_3d_block,
    coarsen_fields_vertical_at_level,
    coarsen_products_vertical_at_level,
    coarsen_dz_profile_2x_at_level,
    coarsen_dz_profile_factor,
    coarsen3d_vertical_mean,
    coarsen_products_at_level,
    coarsen_products_3d_block,
    coarsen_fields_valid_box,
    coarsen_products_valid_box,
    coarsen_fields_valid_box_at_starts,
    coarsen_products_valid_box_at_starts,
    uniform_stride_for_valid_box,
    valid_box_anchor_starts,
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
    build_convolutional_coarsening_triples(
        nx, ny, nz, dh, min_h, dz_ref, max_z;
        square_horizontal=true,
    ) -> Vector{NTuple{3,Int}}

All valid `(fx, fy, fz)` such that the native grid divides evenly, horizontal blocks respect
`min_h` (via `dh * fx` when `fx > 1`), and nominal vertical thickness `fz * dz_ref <= max_z`.

Unlike [`build_horizontal_levels`](@ref) combined with binary vertical ladders, this enumerates
**independent** 3D block factors (e.g. `3×3×2`), giving more resolution steps in physical space.
Each triple is meant to be applied directly from **native** fields (no reuse across triples).
"""
function build_convolutional_coarsening_triples(
    nx::Int,
    ny::Int,
    nz::Int,
    dh::FT,
    min_h::FT,
    dz_ref::FT,
    max_z::FT;
    square_horizontal::Bool = true,
)::Vector{NTuple{3,Int}} where {FT <: Real}
    triples = NTuple{3,Int}[]
    if square_horizontal
        for fx in 1:nx
            mod(nx, fx) != 0 && continue
            mod(ny, fx) != 0 && continue
            if fx > 1 && dh * fx < min_h
                continue
            end
            for fz in 1:nz
                mod(nz, fz) != 0 && continue
                if fz * dz_ref > max_z
                    continue
                end
                push!(triples, (fx, fx, fz))
            end
        end
    else
        for fx in 1:nx, fy in 1:ny
            mod(nx, fx) != 0 && continue
            mod(ny, fy) != 0 && continue
            if max(fx, fy) > 1 && dh * max(fx, fy) < min_h
                continue
            end
            for fz in 1:nz
                mod(nz, fz) != 0 && continue
                if fz * dz_ref > max_z
                    continue
                end
                push!(triples, (fx, fy, fz))
            end
        end
    end
    unique!(triples)
    sort!(triples; by = t -> (t[3], t[1], t[2]))
    return triples
end

"""
    coarsen_fields_3d_block(fields, fx, fy, fz)

Coarsen every 3D field in the `NamedTuple` with [`ArrayUtils.conv3d_block_mean`](@ref).
"""
function coarsen_fields_3d_block(
    fields::NamedTuple{FN, FV},
    fx::Int,
    fy::Int,
    fz::Int,
) where {FN, FV <: Tuple}
    vals = map(arr -> Array(conv3d_block_mean(arr, fx, fy, fz)), values(fields))
    return NamedTuple{FN}(vals)
end

"""
    coarsen_products_3d_block(fields, product_pairs, fx, fy, fz)

Block-average of `x .* y` over each `fx×fy×fz` cell (same numerics as coarsening `<xy>` under 3D means).
"""
function coarsen_products_3d_block(
    fields::NamedTuple{FN, FV},
    product_pairs::NamedTuple{PN, PV},
    fx::Int,
    fy::Int,
    fz::Int,
) where {FN, FV <: Tuple, PN, PV <: Tuple}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        x = _container_get_field(fields, _field_name_key(x_name))
        y = _container_get_field(fields, _field_name_key(y_name))
        size(x) == size(y) || throw(DimensionMismatch("Fields for product $(out_name) have mismatched sizes"))

        nx, ny, nz = size(x)
        nxo = div(nx, fx)
        nyo = div(ny, fy)
        nzo = div(nz, fz)
        T = eltype(x)
        inv_vol = one(T) / T(fx * fy * fz)
        prod_coarse = similar(x, nxo, nyo, nzo)

        @inbounds for k in 1:nzo
            base_k = (k - 1) * fz + 1
            for j in 1:nyo
                base_j = (j - 1) * fy + 1
                for i in 1:nxo
                    base_i = (i - 1) * fx + 1
                    acc = zero(T)
                    for kk in 0:(fz - 1)
                        z = base_k + kk
                        for jj in 0:(fy - 1)
                            yidx = base_j + jj
                            for ii in 0:(fx - 1)
                                xi = base_i + ii
                                acc += x[xi, yidx, z] * y[xi, yidx, z]
                            end
                        end
                    end
                    prod_coarse[i, j, k] = acc * inv_vol
                end
            end
        end
        prod_coarse
    end
    return NamedTuple{PN}(vals)
end

"""
    coarsen_fields_valid_box(fields, wx, wy, wz; stride_h=1, stride_v=1, stride_z=1)

Valid sliding 3D box mean for every field in `fields` (same numerics as [`ArrayUtils.conv3d_valid_box_mean`](@ref)).
"""
function coarsen_fields_valid_box(
    fields::NamedTuple{FN, FV},
    wx::Int,
    wy::Int,
    wz::Int;
    stride_h::Int = 1,
    stride_v::Int = 1,
    stride_z::Int = 1,
) where {FN, FV <: Tuple}
    vals = map(arr -> Array(conv3d_valid_box_mean(arr, wx, wy, wz; stride_h, stride_v, stride_z)), values(fields))
    return NamedTuple{FN}(vals)
end

function coarsen_fields_valid_box_at_starts(
    fields::NamedTuple{FN, FV},
    wx::Int,
    wy::Int,
    wz::Int,
    ix::AbstractVector{Int},
    iy::AbstractVector{Int},
    iz::AbstractVector{Int},
) where {FN, FV <: Tuple}
    vals = map(arr -> Array(conv3d_valid_box_mean_at_starts(arr, wx, wy, wz, ix, iy, iz)), values(fields))
    return NamedTuple{FN}(vals)
end

"""
    coarsen_products_valid_box(fields, product_pairs, wx, wy, wz; stride_h=1, stride_v=1, stride_z=1)

Sliding local mean of `x .* y` over each valid `wx×wy×wz` window.
"""
function coarsen_products_valid_box(
    fields::NamedTuple{FN, FV},
    product_pairs::NamedTuple{PN, PV},
    wx::Int,
    wy::Int,
    wz::Int;
    stride_h::Int = 1,
    stride_v::Int = 1,
    stride_z::Int = 1,
) where {FN, FV <: Tuple, PN, PV <: Tuple}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        x = _container_get_field(fields, _field_name_key(x_name))
        y = _container_get_field(fields, _field_name_key(y_name))
        size(x) == size(y) || throw(DimensionMismatch("Fields for product $(out_name) have mismatched sizes"))
        Array(conv3d_valid_box_product_mean(x, y, wx, wy, wz; stride_h, stride_v, stride_z))
    end
    return NamedTuple{PN}(vals)
end

function coarsen_products_valid_box_at_starts(
    fields::NamedTuple{FN, FV},
    product_pairs::NamedTuple{PN, PV},
    wx::Int,
    wy::Int,
    wz::Int,
    ix::AbstractVector{Int},
    iy::AbstractVector{Int},
    iz::AbstractVector{Int},
) where {FN, FV <: Tuple, PN, PV <: Tuple}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        x = _container_get_field(fields, _field_name_key(x_name))
        y = _container_get_field(fields, _field_name_key(y_name))
        size(x) == size(y) || throw(DimensionMismatch("Fields for product $(out_name) have mismatched sizes"))
        Array(conv3d_valid_box_product_mean_at_starts(x, y, wx, wy, wz, ix, iy, iz))
    end
    return NamedTuple{PN}(vals)
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
