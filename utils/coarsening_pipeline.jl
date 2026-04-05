module CoarseningPipeline

using Statistics: Statistics

include("array_utils.jl")
using .ArrayUtils
using ..StatisticalMethods: StatisticalMethods

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
    build_horizontal_multilevel_views,
    coarsen_products_moments_horizontal_native,
    coarsen_moments_horizontal_merge,
    coarsen_moments_vertical_merge,
    coarsen_products_moments_3d_block

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
        Acc = ArrayUtils._block_reduction_accum_type(T)
        inv_vol = one(Acc) / Acc(fx * fy * fz)
        prod_coarse = similar(x, nxo, nyo, nzo)

        @inbounds for k in 1:nzo
            base_k = (k - 1) * fz + 1
            for j in 1:nyo
                base_j = (j - 1) * fy + 1
                for i in 1:nxo
                    base_i = (i - 1) * fx + 1
                    acc = zero(Acc)
                    for kk in 0:(fz - 1)
                        z = base_k + kk
                        for jj in 0:(fy - 1)
                            yidx = base_j + jj
                            for ii in 0:(fx - 1)
                                xi = base_i + ii
                                acc += Acc(x[xi, yidx, z]) * Acc(y[xi, yidx, z])
                            end
                        end
                    end
                    prod_coarse[i, j, k] = T(acc * inv_vol)
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
        
        Acc = ArrayUtils._block_reduction_accum_type(T)
        inv_area = one(Acc) / Acc(fx * fy)
        @inbounds for k in 1:nz
            for j in 1:nyo
                base_j = (j - 1) * fy + 1
                for i in 1:nxo
                    base_i = (i - 1) * fx + 1
                    acc = zero(Acc)
                    for jj in 0:(fy - 1)
                        for ii in 0:(fx - 1)
                            acc += Acc(x[base_i + ii, base_j + jj, k]) * Acc(y[base_i + ii, base_j + jj, k])
                        end
                    end
                    prod_coarse[i, j, k] = T(acc * inv_area)
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
        Acc = ArrayUtils._block_reduction_accum_type(T)
        inv_area = one(Acc) / Acc(fx * fy)
        @inbounds for k in 1:nz
            for j in 1:nyo
                base_j = (j - 1) * fy + 1
                for i in 1:nxo
                    base_i = (i - 1) * fx + 1
                    acc = zero(Acc)
                    for jj in 0:(fy - 1)
                        for ii in 0:(fx - 1)
                            acc += Acc(x[base_i + ii, base_j + jj, k]) * Acc(y[base_i + ii, base_j + jj, k])
                        end
                    end
                    prod_coarse[i, j, k] = T(acc * inv_area)
                end
            end
        end
        prod_coarse
    end
    return NamedTuple{PN}(vals)
end

function _coarsen3d_horizontal_merge_M2!(
    out_M2::AbstractArray{T, 3},
    mu::AbstractArray{T, 3},
    M2::AbstractArray{T, 3},
    fh::Int,
    fv::Int,
    n_each::Int,
) where {T <: AbstractFloat}
    nx, ny, nz = size(mu)
    size(M2) == (nx, ny, nz) || throw(DimensionMismatch("M2 shape"))
    nxo, nyo = div(nx, fh), div(ny, fv)
    size(out_M2) == (nxo, nyo, nz) || throw(DimensionMismatch("out_M2 shape"))
    kblk = fh * fv
    # Wider scratch for Chan merge (same policy as `_block_reduction_accum_type` in array_utils).
    Acc = ArrayUtils._block_reduction_accum_type(T)
    buf_μ = Vector{Acc}(undef, kblk)
    buf_M2 = Vector{Acc}(undef, kblk)
    @inbounds for k in 1:nz
        for j in 1:nyo
            for i in 1:nxo
                t = 1
                for jj in 0:(fv - 1)
                    for ii in 0:(fh - 1)
                        buf_μ[t] = Acc(mu[(i - 1) * fh + ii + 1, (j - 1) * fv + jj + 1, k])
                        buf_M2[t] = Acc(M2[(i - 1) * fh + ii + 1, (j - 1) * fv + jj + 1, k])
                        t += 1
                    end
                end
                _, _, M2p = StatisticalMethods.merge_variance_children(n_each, buf_μ, buf_M2)
                out_M2[i, j, k] = T(M2p)
            end
        end
    end
    return out_M2
end

function _coarsen3d_vertical_merge_M2!(
    out_M2::AbstractArray{T, 3},
    mu::AbstractArray{T, 3},
    M2::AbstractArray{T, 3},
    fz::Int,
    n_each::Int,
) where {T <: AbstractFloat}
    nx, ny, nz = size(mu)
    nzo = div(nz, fz)
    size(out_M2) == (nx, ny, nzo) || throw(DimensionMismatch("out_M2 shape"))
    Acc = ArrayUtils._block_reduction_accum_type(T)
    buf_μ = Vector{Acc}(undef, fz)
    buf_M2 = Vector{Acc}(undef, fz)
    @inbounds for k in 1:nzo
        base_k = (k - 1) * fz + 1
        for j in 1:ny
            for i in 1:nx
                for kk in 0:(fz - 1)
                    buf_μ[kk + 1] = Acc(mu[i, j, base_k + kk])
                    buf_M2[kk + 1] = Acc(M2[i, j, base_k + kk])
                end
                _, _, M2p = StatisticalMethods.merge_variance_children(n_each, buf_μ, buf_M2)
                out_M2[i, j, k] = T(M2p)
            end
        end
    end
    return out_M2
end

function _coarsen3d_horizontal_merge_C!(
    out_C::AbstractArray{T, 3},
    mux::AbstractArray{T, 3},
    muy::AbstractArray{T, 3},
    C::AbstractArray{T, 3},
    fh::Int,
    fv::Int,
    n_each::Int,
) where {T <: AbstractFloat}
    nx, ny, nz = size(mux)
    size(muy) == (nx, ny, nz) || throw(DimensionMismatch("muy shape"))
    size(C) == (nx, ny, nz) || throw(DimensionMismatch("C shape"))
    nxo, nyo = div(nx, fh), div(ny, fv)
    size(out_C) == (nxo, nyo, nz) || throw(DimensionMismatch("out_C shape"))
    kblk = fh * fv
    Acc = ArrayUtils._block_reduction_accum_type(T)
    buf_x = Vector{Acc}(undef, kblk)
    buf_y = Vector{Acc}(undef, kblk)
    buf_C = Vector{Acc}(undef, kblk)
    @inbounds for k in 1:nz
        for j in 1:nyo
            for i in 1:nxo
                t = 1
                for jj in 0:(fv - 1)
                    for ii in 0:(fh - 1)
                        ix = (i - 1) * fh + ii + 1
                        iy = (j - 1) * fv + jj + 1
                        buf_x[t] = Acc(mux[ix, iy, k])
                        buf_y[t] = Acc(muy[ix, iy, k])
                        buf_C[t] = Acc(C[ix, iy, k])
                        t += 1
                    end
                end
                _, _, _, Cp = StatisticalMethods.merge_covariance_children(n_each, buf_x, buf_y, buf_C)
                out_C[i, j, k] = T(Cp)
            end
        end
    end
    return out_C
end

function _coarsen3d_vertical_merge_C!(
    out_C::AbstractArray{T, 3},
    mux::AbstractArray{T, 3},
    muy::AbstractArray{T, 3},
    C::AbstractArray{T, 3},
    fz::Int,
    n_each::Int,
) where {T <: AbstractFloat}
    nx, ny, nz = size(mux)
    nzo = div(nz, fz)
    size(out_C) == (nx, ny, nzo) || throw(DimensionMismatch("out_C shape"))
    Acc = ArrayUtils._block_reduction_accum_type(T)
    buf_x = Vector{Acc}(undef, fz)
    buf_y = Vector{Acc}(undef, fz)
    buf_C = Vector{Acc}(undef, fz)
    @inbounds for k in 1:nzo
        base_k = (k - 1) * fz + 1
        for j in 1:ny
            for i in 1:nx
                for kk in 0:(fz - 1)
                    buf_x[kk + 1] = Acc(mux[i, j, base_k + kk])
                    buf_y[kk + 1] = Acc(muy[i, j, base_k + kk])
                    buf_C[kk + 1] = Acc(C[i, j, base_k + kk])
                end
                _, _, _, Cp = StatisticalMethods.merge_covariance_children(n_each, buf_x, buf_y, buf_C)
                out_C[i, j, k] = T(Cp)
            end
        end
    end
    return out_C
end

"""
    coarsen_products_moments_horizontal_native(fields, product_pairs, nh)

Horizontal `nh×nh` block stats at native vertical resolution: self-pairs return **M2** (sum of squared
deviations); cross-pairs return **C** (sum of `(x−x̄)(y−ȳ)` within each block).
"""
function coarsen_products_moments_horizontal_native(
    fields::NamedTuple{FN, FV},
    product_pairs::NamedTuple{PN, PV},
    nh::Int,
) where {FN, FV <: Tuple, PN, PV <: Tuple}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        x = _container_get_field(fields, _field_name_key(x_name))
        y = _container_get_field(fields, _field_name_key(y_name))
        size(x) == size(y) || throw(DimensionMismatch("Fields for product $(out_name) have mismatched sizes"))
        if x_name == y_name
            _, M2 = ArrayUtils.conv3d_block_mean_M2(x, nh, nh, 1)
            M2
        else
            _, _, C = ArrayUtils.conv3d_block_covariance_C(x, y, nh, nh, 1)
            C
        end
    end
    return NamedTuple{PN}(vals)
end

"""
    coarsen_moments_horizontal_merge(means_src, moments_src, product_pairs, fh, fv, n_each)

Chan/Pebay merge of child **M2** / **C** fields over `fh×fv` horizontal blocks. Child voxels each
represent `n_each` samples.
"""
function coarsen_moments_horizontal_merge(
    means_src::NamedTuple,
    moments_src::NamedTuple{PN},
    product_pairs::NamedTuple{PN},
    fh::Int,
    fv::Int,
    n_each::Int,
) where {PN}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        if x_name == y_name
            μ = _container_get_field(means_src, _field_name_key(x_name))
            M2c = getproperty(moments_src, out_name)
            out = similar(μ, div(size(μ, 1), fh), div(size(μ, 2), fv), size(μ, 3))
            _coarsen3d_horizontal_merge_M2!(out, μ, M2c, fh, fv, n_each)
        else
            mux = _container_get_field(means_src, _field_name_key(x_name))
            muy = _container_get_field(means_src, _field_name_key(y_name))
            Cc = getproperty(moments_src, out_name)
            out = similar(mux, div(size(mux, 1), fh), div(size(mux, 2), fv), size(mux, 3))
            _coarsen3d_horizontal_merge_C!(out, mux, muy, Cc, fh, fv, n_each)
        end
    end
    return NamedTuple{PN}(vals)
end

"""
    coarsen_moments_vertical_merge(means_src, moments_src, product_pairs, fz, n_each)

Vertical Chan/Pebay merge along `z` in groups of `fz`. Each fine voxel represents `n_each` samples.
"""
function coarsen_moments_vertical_merge(
    means_src::NamedTuple,
    moments_src::NamedTuple{PN},
    product_pairs::NamedTuple{PN},
    fz::Int,
    n_each::Int,
) where {PN}
    vals = map(PN) do out_name
        x_name, y_name = getproperty(product_pairs, out_name)
        if x_name == y_name
            μ = _container_get_field(means_src, _field_name_key(x_name))
            M2c = getproperty(moments_src, out_name)
            out = similar(μ, size(μ, 1), size(μ, 2), div(size(μ, 3), fz))
            _coarsen3d_vertical_merge_M2!(out, μ, M2c, fz, n_each)
        else
            mux = _container_get_field(means_src, _field_name_key(x_name))
            muy = _container_get_field(means_src, _field_name_key(y_name))
            Cc = getproperty(moments_src, out_name)
            out = similar(mux, size(mux, 1), size(mux, 2), div(size(mux, 3), fz))
            _coarsen3d_vertical_merge_C!(out, mux, muy, Cc, fz, n_each)
        end
    end
    return NamedTuple{PN}(vals)
end

"""
    coarsen_products_moments_3d_block(fields, product_pairs, fx, fy, fz)

Full 3D block **M2** (self pairs) or **C** (cross pairs) from native fields (non-square tower path).
"""
function coarsen_products_moments_3d_block(
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
        if x_name == y_name
            _, M2 = ArrayUtils.conv3d_block_mean_M2(x, fx, fy, fz)
            M2
        else
            _, _, C = ArrayUtils.conv3d_block_covariance_C(x, y, fx, fy, fz)
            C
        end
    end
    return NamedTuple{PN}(vals)
end

"""
    covariance_from_moments(mean_xy, mean_x, mean_y)

Thin wrapper around [`StatisticalMethods.covariance_from_moments`](@ref) for callers that only load
`CoarseningPipeline`.
"""
function covariance_from_moments(
    mean_xy::AbstractArray{T, 3},
    mean_x::AbstractArray{T, 3},
    mean_y::AbstractArray{T, 3},
) where {T <: Real}
    return StatisticalMethods.covariance_from_moments(mean_xy, mean_x, mean_y)
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
