module DatasetBuilderImpl

using DataFrames: DataFrames
using ..DatasetBuilder: DatasetBuilder
using ..CoarseGraining: CoarseGraining
using ..Dynamics: Dynamics

include("coarsening_pipeline.jl")
using .CoarseningPipeline

export process_abstract_chunk_impl

@inline function _as_f32_array3(x::Array{Float32, 3})
    return x
end

@inline function _as_f32_array3(x::AbstractArray{<:Real, 3})
    return Float32.(x)
end

@inline function _covariance_from_moments!(out::AbstractArray{T, 3}, mean_xy::AbstractArray{T, 3}, mean_x::AbstractArray{T, 3}, mean_y::AbstractArray{T, 3}) where {T <: Real}
    @inbounds for idx in eachindex(out)
        out[idx] = mean_xy[idx] - mean_x[idx] * mean_y[idx]
    end
    return out
end

@inline function _tke_from_moments!(out::AbstractArray{T, 3}, mean_sq_u::AbstractArray{T, 3}, mean_u::AbstractArray{T, 3}, mean_sq_v::AbstractArray{T, 3}, mean_v::AbstractArray{T, 3}, mean_sq_w::AbstractArray{T, 3}, mean_w::AbstractArray{T, 3}) where {T <: Real}
    half = T(0.5)
    @inbounds for idx in eachindex(out)
        var_u = mean_sq_u[idx] - mean_u[idx] * mean_u[idx]
        var_v = mean_sq_v[idx] - mean_v[idx] * mean_v[idx]
        var_w = mean_sq_w[idx] - mean_w[idx] * mean_w[idx]
        out[idx] = half * (var_u + var_v + var_w)
    end
    return out
end

@inline function _all_finite_emitted_diagnostics(
    i::Int,
    j::Int,
    k::Int,
    v_qt,
    v_h,
    v_ta,
    v_p,
    v_rho,
    v_w,
    v_ql,
    v_qi,
    v_liq_fraction,
    v_ice_fraction,
    v_cloud_fraction,
    tke,
    var_qt,
    var_ql,
    var_qi,
    var_w,
    var_h,
    cov_qt_ql,
    cov_qt_qi,
    cov_qt_w,
    cov_qt_h,
    cov_ql_qi,
    cov_ql_w,
    cov_ql_h,
    cov_qi_w,
    cov_qi_h,
    cov_w_h,
)::Bool
    @inbounds return (
        isfinite(v_qt[i, j, k]) &&
        isfinite(v_h[i, j, k]) &&
        isfinite(v_ta[i, j, k]) &&
        isfinite(v_p[i, j, k]) &&
        isfinite(v_rho[i, j, k]) &&
        isfinite(v_w[i, j, k]) &&
        isfinite(v_ql[i, j, k]) &&
        isfinite(v_qi[i, j, k]) &&
        isfinite(v_liq_fraction[i, j, k]) &&
        isfinite(v_ice_fraction[i, j, k]) &&
        isfinite(v_cloud_fraction[i, j, k]) &&
        isfinite(tke[i, j, k]) &&
        isfinite(var_qt[i, j, k]) &&
        isfinite(var_ql[i, j, k]) &&
        isfinite(var_qi[i, j, k]) &&
        isfinite(var_w[i, j, k]) &&
        isfinite(var_h[i, j, k]) &&
        isfinite(cov_qt_ql[i, j, k]) &&
        isfinite(cov_qt_qi[i, j, k]) &&
        isfinite(cov_qt_w[i, j, k]) &&
        isfinite(cov_qt_h[i, j, k]) &&
        isfinite(cov_ql_qi[i, j, k]) &&
        isfinite(cov_ql_w[i, j, k]) &&
        isfinite(cov_ql_h[i, j, k]) &&
        isfinite(cov_qi_w[i, j, k]) &&
        isfinite(cov_qi_h[i, j, k]) &&
        isfinite(cov_w_h[i, j, k])
    )
end

# Allocation free
const PRODUCT_PAIRS = (
    qt_qt = (:hus, :hus),
    ql_ql = (:clw, :clw),
    qi_qi = (:cli, :cli),
    u_u   = (:ua, :ua),
    v_v   = (:va, :va),
    w_w   = (:wa, :wa),
    h_h   = (:thetali, :thetali),
    qt_ql = (:hus, :clw),
    qt_qi = (:hus, :cli),
    qt_w  = (:hus, :wa),
    qt_h  = (:hus, :thetali),
    ql_qi = (:clw, :cli),
    ql_w  = (:clw, :wa),
    ql_h  = (:clw, :thetali),
    qi_w  = (:cli, :wa),
    qi_h  = (:cli, :thetali),
    w_h   = (:wa, :thetali),
)

"""
        process_abstract_chunk_impl(fine_fields, metadata, spatial_info, max_dz=400f0)

Build one tabular training chunk from fine-resolution 3D LES fields.

# Inputs

- `fine_fields`: map of canonical field name -> 3D array `(x, y, z)`.
    Required keys: `hus`, `thetali`, `ta`, `pfull`, `rhoa`, `ua`, `va`, `wa`,
    `clw`, `cli`.
- `metadata`: run/case metadata attached to emitted rows.
    Can be either an `AbstractDict{Symbol}` or a named tuple with matching keys.
- `spatial_info`: grid metadata. Required fields: `dx_native`, `domain_h`,
    `dz_native_profile`. Optional fields: `min_h_resolution`, `seeds_h`,
    `coarsening_mode` (`:binary` default, or `:convolutional` for 3D block means at
    many `(fx,fy,fz)` triples via [`CoarseningPipeline.build_convolutional_coarsening_triples`](@ref)),
    `convolutional_square_horizontal` (default `true`), `convolutional_triples` (optional explicit
    `Vector{NTuple{3,Int}}` overriding the auto schedule).
- `max_dz`: maximum effective vertical spacing used to build the vertical
    coarsening scheme.

# Algorithm

1. Derive cloud-presence indicator fields from `clw` and `cli`.
2. Build horizontal coarsening levels and iteratively coarsen means/products.
3. For each horizontal level, apply vertical coarsening stages.
4. At each stage, compute moments/covariances/TKE and construct a drop mask
     (clear-air, dropped-z, non-finite).
5. Flatten surviving cells into schema-ordered table rows.

# Notes on Performance

- The function is designed for type-stable metadata/spatial-info access.
- Coarsening state is reused across horizontal levels (`means_prev/prods_prev`)
    to avoid recomputing from native resolution each iteration.
- Vertical coarsening schedule `z_schemes` is computed once per chunk (not per horizontal level).
- Drop masks (clear-air, z-prune, non-finite) are fused into a single pass over a **reused** `BitArray` slab (`view` per z-stage size).
- `identify_empty_z_levels_from_ql_qi` avoids a `ql .+ qi` temporary for vertical pruning.
- `tke` / variance / covariance slabs are **preallocated once per horizontal level** and filled via `view`s (no per-z-stage `similar`).
- `flatten_and_filter!` does one column-major gather pass (see `dataset_builder.jl`).
"""
function process_abstract_chunk_impl(
    fine_fields::AbstractDict{String, <:AbstractArray{FT}},
    metadata,
    spatial_info;
    max_dz::FT = FT(400.0f0),
) where {FT <: Real}
    # 1) Required fine-resolution base fields.
    base_qt = fine_fields["hus"]
    base_h = fine_fields["thetali"]
    base_ta = fine_fields["ta"]
    base_p = fine_fields["pfull"]
    base_rho = fine_fields["rhoa"]
    base_u = fine_fields["ua"]
    base_v = fine_fields["va"]
    base_w = fine_fields["wa"]
    base_ql = fine_fields["clw"]
    base_qi = fine_fields["cli"]

    cloud_threshold = DatasetBuilder.CLOUD_PRESENCE_THRESHOLD

    # 2) Presence indicators used downstream for cloud fractions.
    base_liq_presence = similar(base_ql)
    base_ice_presence = similar(base_qi)
    base_cloud_presence = similar(base_ql)
    @inbounds for idx in eachindex(base_ql)
        liq_01 = DatasetBuilder._indicator_01(base_ql[idx], cloud_threshold)
        ice_01 = DatasetBuilder._indicator_01(base_qi[idx], cloud_threshold)
        base_liq_presence[idx] = liq_01
        base_ice_presence[idx] = ice_01
        base_cloud_presence[idx] = (liq_01 > zero(FT) || ice_01 > zero(FT)) ? one(FT) : zero(FT)
    end

    # 3) Typed spatial metadata extraction (supports Dict or NamedTuple).
    dx_native = FT(_spatial_required(spatial_info, :dx_native))
    min_h_resolution = FT(_spatial_lookup(spatial_info, :min_h_resolution, 1000.0f0))
    seeds_h = _spatial_lookup(spatial_info, :seeds_h, (1,))

    # 4) Canonical field dictionary consumed by coarsening kernels.
    fields = (
        hus = base_qt,
        thetali = base_h,
        ta = base_ta,
        pfull = base_p,
        rhoa = base_rho,
        ua = base_u,
        va = base_v,
        wa = base_w,
        clw = base_ql,
        cli = base_qi,
        liq_fraction = base_liq_presence,
        ice_fraction = base_ice_presence,
        cloud_fraction = base_cloud_presence,
    )

    product_pairs = PRODUCT_PAIRS

    # 5) Horizontal schedule + vertical scheme (latter is independent of horizontal level).
    out_acc = nothing
    domain_h = FT(_spatial_required(spatial_info, :domain_h))
    dz_native = _spatial_required(spatial_info, :dz_native_profile)
    nx, ny, nz = size(base_qt)

    coarsening_mode = _spatial_lookup(spatial_info, :coarsening_mode, :binary)
    if coarsening_mode === :convolutional
        return _process_abstract_chunk_convolutional(
            fields,
            product_pairs,
            metadata,
            spatial_info,
            max_dz,
            cloud_threshold,
            domain_h,
            dz_native,
            nx,
            ny,
            nz,
            FT,
        )
    end

    z_schemes = CoarseGraining.compute_z_coarsening_scheme(dz_native, max_dz)
    z_scheme_factors = Int[s[1] for s in z_schemes]

    h_levels = build_horizontal_levels(
        nx,
        ny,
        dx_native;
        seeds=seeds_h,
        min_h=min_h_resolution,
        include_full_domain=false,
    )

    means_prev = nothing
    prods_prev = nothing
    prev_factor = 0

    # Horizontal multiscale sweep.
    for h_level in h_levels
        factor = h_level.fx
        current_resolution_h = FT(h_level.resolution_h)

        # Reuse previous level outputs when possible instead of recoarsening
        # directly from native fields each time.
        means = if prev_factor == 0
            coarsen_fields_at_level(fields, factor, factor)
        else
            ratio = div(factor, prev_factor)
            coarsen_fields_at_level(means_prev, ratio, ratio)
        end

        prods = if prev_factor == 0
            coarsen_products_at_level(fields, product_pairs, factor, factor)
        else
            ratio = div(factor, prev_factor)
            coarsen_fields_at_level(prods_prev, ratio, ratio)
        end

        means_prev = means
        prods_prev = prods
        prev_factor = factor

        c_qt = _field_get(means, :hus)
        c_h = _field_get(means, :thetali)
        c_ta = _field_get(means, :ta)
        c_p = _field_get(means, :pfull)
        c_rho = _field_get(means, :rhoa)
        c_u = _field_get(means, :ua)
        c_v = _field_get(means, :va)
        c_w = _field_get(means, :wa)
        c_ql = _field_get(means, :clw)
        c_qi = _field_get(means, :cli)
        c_liq_fraction = _field_get(means, :liq_fraction)
        c_ice_fraction = _field_get(means, :ice_fraction)
        c_cloud_fraction = _field_get(means, :cloud_fraction)

        any_cloud = false
        @inbounds for idx in eachindex(c_ql)
            any_cloud |= (c_ql[idx] + c_qi[idx]) >= cloud_threshold
        end
        !any_cloud && break

        c_prod_qt_qt = _field_get(prods, :qt_qt)
        c_prod_ql_ql = _field_get(prods, :ql_ql)
        c_prod_qi_qi = _field_get(prods, :qi_qi)
        c_prod_u_u = _field_get(prods, :u_u)
        c_prod_v_v = _field_get(prods, :v_v)
        c_prod_w_w = _field_get(prods, :w_w)
        c_prod_h_h = _field_get(prods, :h_h)
        c_prod_qt_ql = _field_get(prods, :qt_ql)
        c_prod_qt_qi = _field_get(prods, :qt_qi)
        c_prod_qt_w = _field_get(prods, :qt_w)
        c_prod_qt_h = _field_get(prods, :qt_h)
        c_prod_ql_qi = _field_get(prods, :ql_qi)
        c_prod_ql_w = _field_get(prods, :ql_w)
        c_prod_ql_h = _field_get(prods, :ql_h)
        c_prod_qi_w = _field_get(prods, :qi_w)
        c_prod_qi_h = _field_get(prods, :qi_h)
        c_prod_w_h = _field_get(prods, :w_h)

        v_qt = c_qt
        v_h = c_h
        v_ta = c_ta
        v_p = c_p
        v_rho = c_rho
        v_u = c_u
        v_v = c_v
        v_w = c_w
        v_ql = c_ql
        v_qi = c_qi
        v_liq_fraction = c_liq_fraction
        v_ice_fraction = c_ice_fraction
        v_cloud_fraction = c_cloud_fraction

        v_prod_qt_qt = c_prod_qt_qt
        v_prod_ql_ql = c_prod_ql_ql
        v_prod_qi_qi = c_prod_qi_qi
        v_prod_u_u = c_prod_u_u
        v_prod_v_v = c_prod_v_v
        v_prod_w_w = c_prod_w_w
        v_prod_h_h = c_prod_h_h
        v_prod_qt_ql = c_prod_qt_ql
        v_prod_qt_qi = c_prod_qt_qi
        v_prod_qt_w = c_prod_qt_w
        v_prod_qt_h = c_prod_qt_h
        v_prod_ql_qi = c_prod_ql_qi
        v_prod_ql_w = c_prod_ql_w
        v_prod_ql_h = c_prod_ql_h
        v_prod_qi_w = c_prod_qi_w
        v_prod_qi_h = c_prod_qi_h
        v_prod_w_h = c_prod_w_h
        v_dz_profile = dz_native

        # Reusable slabs (max size = first vertical stage at this horizontal level).
        s_diag = size(c_ql)
        diag_tke = Array{FT}(undef, s_diag)
        diag_var_qt = Array{FT}(undef, s_diag)
        diag_var_ql = Array{FT}(undef, s_diag)
        diag_var_qi = Array{FT}(undef, s_diag)
        diag_var_w = Array{FT}(undef, s_diag)
        diag_var_h = Array{FT}(undef, s_diag)
        diag_cov_qt_ql = Array{FT}(undef, s_diag)
        diag_cov_qt_qi = Array{FT}(undef, s_diag)
        diag_cov_qt_w = Array{FT}(undef, s_diag)
        diag_cov_qt_h = Array{FT}(undef, s_diag)
        diag_cov_ql_qi = Array{FT}(undef, s_diag)
        diag_cov_ql_w = Array{FT}(undef, s_diag)
        diag_cov_ql_h = Array{FT}(undef, s_diag)
        diag_cov_qi_w = Array{FT}(undef, s_diag)
        diag_cov_qi_h = Array{FT}(undef, s_diag)
        diag_cov_w_h = Array{FT}(undef, s_diag)
        combined_mask_buf = BitArray(undef, s_diag)

        # Vertical multiscale sweep for the current horizontal level.
        for (z_level_idx, (z_factor, _)) in enumerate(z_schemes)
            nx_z, ny_z, nz_z = size(v_ql)
            ir, jr, kr = Base.OneTo(nx_z), Base.OneTo(ny_z), Base.OneTo(nz_z)
            tke = view(diag_tke, ir, jr, kr)
            _tke_from_moments!(tke, v_prod_u_u, v_u, v_prod_v_v, v_v, v_prod_w_w, v_w)

            var_qt = view(diag_var_qt, ir, jr, kr)
            var_ql = view(diag_var_ql, ir, jr, kr)
            var_qi = view(diag_var_qi, ir, jr, kr)
            var_w = view(diag_var_w, ir, jr, kr)
            var_h = view(diag_var_h, ir, jr, kr)
            _covariance_from_moments!(var_qt, v_prod_qt_qt, v_qt, v_qt)
            _covariance_from_moments!(var_ql, v_prod_ql_ql, v_ql, v_ql)
            _covariance_from_moments!(var_qi, v_prod_qi_qi, v_qi, v_qi)
            _covariance_from_moments!(var_w, v_prod_w_w, v_w, v_w)
            _covariance_from_moments!(var_h, v_prod_h_h, v_h, v_h)

            cov_qt_ql = view(diag_cov_qt_ql, ir, jr, kr)
            cov_qt_qi = view(diag_cov_qt_qi, ir, jr, kr)
            cov_qt_w = view(diag_cov_qt_w, ir, jr, kr)
            cov_qt_h = view(diag_cov_qt_h, ir, jr, kr)
            cov_ql_qi = view(diag_cov_ql_qi, ir, jr, kr)
            cov_ql_w = view(diag_cov_ql_w, ir, jr, kr)
            cov_ql_h = view(diag_cov_ql_h, ir, jr, kr)
            cov_qi_w = view(diag_cov_qi_w, ir, jr, kr)
            cov_qi_h = view(diag_cov_qi_h, ir, jr, kr)
            cov_w_h = view(diag_cov_w_h, ir, jr, kr)
            _covariance_from_moments!(cov_qt_ql, v_prod_qt_ql, v_qt, v_ql)
            _covariance_from_moments!(cov_qt_qi, v_prod_qt_qi, v_qt, v_qi)
            _covariance_from_moments!(cov_qt_w, v_prod_qt_w, v_qt, v_w)
            _covariance_from_moments!(cov_qt_h, v_prod_qt_h, v_qt, v_h)
            _covariance_from_moments!(cov_ql_qi, v_prod_ql_qi, v_ql, v_qi)
            _covariance_from_moments!(cov_ql_w, v_prod_ql_w, v_ql, v_w)
            _covariance_from_moments!(cov_ql_h, v_prod_ql_h, v_ql, v_h)
            _covariance_from_moments!(cov_qi_w, v_prod_qi_w, v_qi, v_w)
            _covariance_from_moments!(cov_qi_h, v_prod_qi_h, v_qi, v_h)
            _covariance_from_moments!(cov_w_h, v_prod_w_h, v_w, v_h)

            empty_z_levels = CoarseGraining.identify_empty_z_levels_from_ql_qi(v_ql, v_qi, cloud_threshold)
            future_z_factors = @view z_scheme_factors[(z_level_idx + 1):end]
            z_keep_mask = CoarseGraining.build_z_level_keep_mask(empty_z_levels, z_factor, future_z_factors)
            any(z_keep_mask) || continue

            # 7) Single fused drop mask (view into reusable BitArray buffer).
            combined_mask = view(combined_mask_buf, ir, jr, kr)
            @inbounds for k in 1:nz_z
                z_drop_k = !z_keep_mask[k]
                for j in 1:ny_z
                    for i in 1:nx_z
                        if z_drop_k
                            combined_mask[i, j, k] = true
                        elseif (v_ql[i, j, k] + v_qi[i, j, k]) < cloud_threshold
                            combined_mask[i, j, k] = true
                        else
                            combined_mask[i, j, k] = !_all_finite_emitted_diagnostics(
                                i,
                                j,
                                k,
                                v_qt,
                                v_h,
                                v_ta,
                                v_p,
                                v_rho,
                                v_w,
                                v_ql,
                                v_qi,
                                v_liq_fraction,
                                v_ice_fraction,
                                v_cloud_fraction,
                                tke,
                                var_qt,
                                var_ql,
                                var_qi,
                                var_w,
                                var_h,
                                cov_qt_ql,
                                cov_qt_qi,
                                cov_qt_w,
                                cov_qt_h,
                                cov_ql_qi,
                                cov_ql_w,
                                cov_ql_h,
                                cov_qi_w,
                                cov_qi_h,
                                cov_w_h,
                            )
                        end
                    end
                end
            end

            # 8) Emit one table for this (h, z) level.
            df_level = DataFrames.DataFrame()
            DatasetBuilder.flatten_and_filter!(
                df_level,
                combined_mask,
                v_qt,
                v_h,
                v_ta,
                v_p,
                v_rho,
                v_w,
                v_ql,
                v_qi,
                v_liq_fraction,
                v_ice_fraction,
                v_cloud_fraction,
                tke,
                var_qt,
                var_ql,
                var_qi,
                var_w,
                var_h,
                cov_qt_ql,
                cov_qt_qi,
                cov_qt_w,
                cov_qt_h,
                cov_ql_qi,
                cov_ql_w,
                cov_ql_h,
                cov_qi_w,
                cov_qi_h,
                cov_w_h,
                v_dz_profile,
                Float32(current_resolution_h),
                domain_h,
                metadata,
            )
            if DataFrames.nrow(df_level) > 0
                if isnothing(out_acc)
                    out_acc = df_level
                else
                    DataFrames.append!(out_acc, df_level)
                end
            end

            # 9) Prepare next vertical level via 2x coarsening.
            if z_level_idx == length(z_schemes) || size(v_qt, 3) < 2
                continue
            end

            v_qt = coarsen3d_vertical_mean(v_qt, 2)
            v_h = coarsen3d_vertical_mean(v_h, 2)
            v_ta = coarsen3d_vertical_mean(v_ta, 2)
            v_p = coarsen3d_vertical_mean(v_p, 2)
            v_rho = coarsen3d_vertical_mean(v_rho, 2)
            v_u = coarsen3d_vertical_mean(v_u, 2)
            v_v = coarsen3d_vertical_mean(v_v, 2)
            v_w = coarsen3d_vertical_mean(v_w, 2)
            v_ql = coarsen3d_vertical_mean(v_ql, 2)
            v_qi = coarsen3d_vertical_mean(v_qi, 2)
            v_liq_fraction = coarsen3d_vertical_mean(v_liq_fraction, 2)
            v_ice_fraction = coarsen3d_vertical_mean(v_ice_fraction, 2)
            v_cloud_fraction = coarsen3d_vertical_mean(v_cloud_fraction, 2)

            v_prod_qt_qt = coarsen3d_vertical_mean(v_prod_qt_qt, 2)
            v_prod_ql_ql = coarsen3d_vertical_mean(v_prod_ql_ql, 2)
            v_prod_qi_qi = coarsen3d_vertical_mean(v_prod_qi_qi, 2)
            v_prod_u_u = coarsen3d_vertical_mean(v_prod_u_u, 2)
            v_prod_v_v = coarsen3d_vertical_mean(v_prod_v_v, 2)
            v_prod_w_w = coarsen3d_vertical_mean(v_prod_w_w, 2)
            v_prod_h_h = coarsen3d_vertical_mean(v_prod_h_h, 2)
            v_prod_qt_ql = coarsen3d_vertical_mean(v_prod_qt_ql, 2)
            v_prod_qt_qi = coarsen3d_vertical_mean(v_prod_qt_qi, 2)
            v_prod_qt_w = coarsen3d_vertical_mean(v_prod_qt_w, 2)
            v_prod_qt_h = coarsen3d_vertical_mean(v_prod_qt_h, 2)
            v_prod_ql_qi = coarsen3d_vertical_mean(v_prod_ql_qi, 2)
            v_prod_ql_w = coarsen3d_vertical_mean(v_prod_ql_w, 2)
            v_prod_ql_h = coarsen3d_vertical_mean(v_prod_ql_h, 2)
            v_prod_qi_w = coarsen3d_vertical_mean(v_prod_qi_w, 2)
            v_prod_qi_h = coarsen3d_vertical_mean(v_prod_qi_h, 2)
            v_prod_w_h = coarsen3d_vertical_mean(v_prod_w_h, 2)

            v_dz_profile = coarsen_dz_profile_2x_at_level(v_dz_profile)
        end
    end

    isnothing(out_acc) && return DataFrames.DataFrame()
    return out_acc
end

function _process_abstract_chunk_convolutional(
    fields::NamedTuple,
    product_pairs::NamedTuple,
    metadata,
    spatial_info,
    max_dz::FT,
    cloud_threshold::Float32,
    domain_h::FT,
    dz_native,
    nx::Int,
    ny::Int,
    nz::Int,
    ::Type{FT},
) where {FT <: Real}
    dx_native = FT(_spatial_required(spatial_info, :dx_native))
    min_h_resolution = FT(_spatial_lookup(spatial_info, :min_h_resolution, 1000.0f0))
    square_h = _spatial_lookup(spatial_info, :convolutional_square_horizontal, true)
    explicit = _spatial_lookup(spatial_info, :convolutional_triples, nothing)

    dz_ref = FT(sum(dz_native) / length(dz_native))

    triples = if explicit === nothing
        build_convolutional_coarsening_triples(
            nx,
            ny,
            nz,
            dx_native,
            min_h_resolution,
            dz_ref,
            FT(max_dz);
            square_horizontal = square_h,
        )
    else
        Vector{NTuple{3,Int}}(collect(explicit))
    end

    future_z_empty = Int[]

    out_acc = nothing
    for (fx, fy, fz) in triples
        means = coarsen_fields_3d_block(fields, fx, fy, fz)
        prods = coarsen_products_3d_block(fields, product_pairs, fx, fy, fz)

        c_qt = _field_get(means, :hus)
        c_h = _field_get(means, :thetali)
        c_ta = _field_get(means, :ta)
        c_p = _field_get(means, :pfull)
        c_rho = _field_get(means, :rhoa)
        c_u = _field_get(means, :ua)
        c_v = _field_get(means, :va)
        c_w = _field_get(means, :wa)
        c_ql = _field_get(means, :clw)
        c_qi = _field_get(means, :cli)
        c_liq_fraction = _field_get(means, :liq_fraction)
        c_ice_fraction = _field_get(means, :ice_fraction)
        c_cloud_fraction = _field_get(means, :cloud_fraction)

        any_cloud = false
        @inbounds for idx in eachindex(c_ql)
            any_cloud |= (c_ql[idx] + c_qi[idx]) >= cloud_threshold
        end
        !any_cloud && continue

        c_prod_qt_qt = _field_get(prods, :qt_qt)
        c_prod_ql_ql = _field_get(prods, :ql_ql)
        c_prod_qi_qi = _field_get(prods, :qi_qi)
        c_prod_u_u = _field_get(prods, :u_u)
        c_prod_v_v = _field_get(prods, :v_v)
        c_prod_w_w = _field_get(prods, :w_w)
        c_prod_h_h = _field_get(prods, :h_h)
        c_prod_qt_ql = _field_get(prods, :qt_ql)
        c_prod_qt_qi = _field_get(prods, :qt_qi)
        c_prod_qt_w = _field_get(prods, :qt_w)
        c_prod_qt_h = _field_get(prods, :qt_h)
        c_prod_ql_qi = _field_get(prods, :ql_qi)
        c_prod_ql_w = _field_get(prods, :ql_w)
        c_prod_ql_h = _field_get(prods, :ql_h)
        c_prod_qi_w = _field_get(prods, :qi_w)
        c_prod_qi_h = _field_get(prods, :qi_h)
        c_prod_w_h = _field_get(prods, :w_h)

        v_qt = c_qt
        v_h = c_h
        v_ta = c_ta
        v_p = c_p
        v_rho = c_rho
        v_u = c_u
        v_v = c_v
        v_w = c_w
        v_ql = c_ql
        v_qi = c_qi
        v_liq_fraction = c_liq_fraction
        v_ice_fraction = c_ice_fraction
        v_cloud_fraction = c_cloud_fraction

        v_prod_qt_qt = c_prod_qt_qt
        v_prod_ql_ql = c_prod_ql_ql
        v_prod_qi_qi = c_prod_qi_qi
        v_prod_u_u = c_prod_u_u
        v_prod_v_v = c_prod_v_v
        v_prod_w_w = c_prod_w_w
        v_prod_h_h = c_prod_h_h
        v_prod_qt_ql = c_prod_qt_ql
        v_prod_qt_qi = c_prod_qt_qi
        v_prod_qt_w = c_prod_qt_w
        v_prod_qt_h = c_prod_qt_h
        v_prod_ql_qi = c_prod_ql_qi
        v_prod_ql_w = c_prod_ql_w
        v_prod_ql_h = c_prod_ql_h
        v_prod_qi_w = c_prod_qi_w
        v_prod_qi_h = c_prod_qi_h
        v_prod_w_h = c_prod_w_h

        v_dz_profile = coarsen_dz_profile_factor(dz_native, fz)
        current_resolution_h = FT(max(fx, fy)) * dx_native

        s_diag = size(c_ql)
        diag_tke = Array{FT}(undef, s_diag)
        diag_var_qt = Array{FT}(undef, s_diag)
        diag_var_ql = Array{FT}(undef, s_diag)
        diag_var_qi = Array{FT}(undef, s_diag)
        diag_var_w = Array{FT}(undef, s_diag)
        diag_var_h = Array{FT}(undef, s_diag)
        diag_cov_qt_ql = Array{FT}(undef, s_diag)
        diag_cov_qt_qi = Array{FT}(undef, s_diag)
        diag_cov_qt_w = Array{FT}(undef, s_diag)
        diag_cov_qt_h = Array{FT}(undef, s_diag)
        diag_cov_ql_qi = Array{FT}(undef, s_diag)
        diag_cov_ql_w = Array{FT}(undef, s_diag)
        diag_cov_ql_h = Array{FT}(undef, s_diag)
        diag_cov_qi_w = Array{FT}(undef, s_diag)
        diag_cov_qi_h = Array{FT}(undef, s_diag)
        diag_cov_w_h = Array{FT}(undef, s_diag)
        combined_mask_buf = BitArray(undef, s_diag)

        nx_z, ny_z, nz_z = size(v_ql)
        ir, jr, kr = Base.OneTo(nx_z), Base.OneTo(ny_z), Base.OneTo(nz_z)
        tke = view(diag_tke, ir, jr, kr)
        _tke_from_moments!(tke, v_prod_u_u, v_u, v_prod_v_v, v_v, v_prod_w_w, v_w)

        var_qt = view(diag_var_qt, ir, jr, kr)
        var_ql = view(diag_var_ql, ir, jr, kr)
        var_qi = view(diag_var_qi, ir, jr, kr)
        var_w = view(diag_var_w, ir, jr, kr)
        var_h = view(diag_var_h, ir, jr, kr)
        _covariance_from_moments!(var_qt, v_prod_qt_qt, v_qt, v_qt)
        _covariance_from_moments!(var_ql, v_prod_ql_ql, v_ql, v_ql)
        _covariance_from_moments!(var_qi, v_prod_qi_qi, v_qi, v_qi)
        _covariance_from_moments!(var_w, v_prod_w_w, v_w, v_w)
        _covariance_from_moments!(var_h, v_prod_h_h, v_h, v_h)

        cov_qt_ql = view(diag_cov_qt_ql, ir, jr, kr)
        cov_qt_qi = view(diag_cov_qt_qi, ir, jr, kr)
        cov_qt_w = view(diag_cov_qt_w, ir, jr, kr)
        cov_qt_h = view(diag_cov_qt_h, ir, jr, kr)
        cov_ql_qi = view(diag_cov_ql_qi, ir, jr, kr)
        cov_ql_w = view(diag_cov_ql_w, ir, jr, kr)
        cov_ql_h = view(diag_cov_ql_h, ir, jr, kr)
        cov_qi_w = view(diag_cov_qi_w, ir, jr, kr)
        cov_qi_h = view(diag_cov_qi_h, ir, jr, kr)
        cov_w_h = view(diag_cov_w_h, ir, jr, kr)
        _covariance_from_moments!(cov_qt_ql, v_prod_qt_ql, v_qt, v_ql)
        _covariance_from_moments!(cov_qt_qi, v_prod_qt_qi, v_qt, v_qi)
        _covariance_from_moments!(cov_qt_w, v_prod_qt_w, v_qt, v_w)
        _covariance_from_moments!(cov_qt_h, v_prod_qt_h, v_qt, v_h)
        _covariance_from_moments!(cov_ql_qi, v_prod_ql_qi, v_ql, v_qi)
        _covariance_from_moments!(cov_ql_w, v_prod_ql_w, v_ql, v_w)
        _covariance_from_moments!(cov_ql_h, v_prod_ql_h, v_ql, v_h)
        _covariance_from_moments!(cov_qi_w, v_prod_qi_w, v_qi, v_w)
        _covariance_from_moments!(cov_qi_h, v_prod_qi_h, v_qi, v_h)
        _covariance_from_moments!(cov_w_h, v_prod_w_h, v_w, v_h)

        empty_z_levels = CoarseGraining.identify_empty_z_levels_from_ql_qi(v_ql, v_qi, cloud_threshold)
        z_keep_mask = CoarseGraining.build_z_level_keep_mask(empty_z_levels, fz, future_z_empty)
        any(z_keep_mask) || continue

        combined_mask = view(combined_mask_buf, ir, jr, kr)
        @inbounds for k in 1:nz_z
            z_drop_k = !z_keep_mask[k]
            for j in 1:ny_z
                for i in 1:nx_z
                    if z_drop_k
                        combined_mask[i, j, k] = true
                    elseif (v_ql[i, j, k] + v_qi[i, j, k]) < cloud_threshold
                        combined_mask[i, j, k] = true
                    else
                        combined_mask[i, j, k] = !_all_finite_emitted_diagnostics(
                            i,
                            j,
                            k,
                            v_qt,
                            v_h,
                            v_ta,
                            v_p,
                            v_rho,
                            v_w,
                            v_ql,
                            v_qi,
                            v_liq_fraction,
                            v_ice_fraction,
                            v_cloud_fraction,
                            tke,
                            var_qt,
                            var_ql,
                            var_qi,
                            var_w,
                            var_h,
                            cov_qt_ql,
                            cov_qt_qi,
                            cov_qt_w,
                            cov_qt_h,
                            cov_ql_qi,
                            cov_ql_w,
                            cov_ql_h,
                            cov_qi_w,
                            cov_qi_h,
                            cov_w_h,
                        )
                    end
                end
            end
        end

        df_level = DataFrames.DataFrame()
        DatasetBuilder.flatten_and_filter!(
            df_level,
            combined_mask,
            v_qt,
            v_h,
            v_ta,
            v_p,
            v_rho,
            v_w,
            v_ql,
            v_qi,
            v_liq_fraction,
            v_ice_fraction,
            v_cloud_fraction,
            tke,
            var_qt,
            var_ql,
            var_qi,
            var_w,
            var_h,
            cov_qt_ql,
            cov_qt_qi,
            cov_qt_w,
            cov_qt_h,
            cov_ql_qi,
            cov_ql_w,
            cov_ql_h,
            cov_qi_w,
            cov_qi_h,
            cov_w_h,
            v_dz_profile,
            Float32(current_resolution_h),
            domain_h,
            metadata,
        )
        if DataFrames.nrow(df_level) > 0
            if isnothing(out_acc)
                out_acc = df_level
            else
                DataFrames.append!(out_acc, df_level)
            end
        end
    end

    isnothing(out_acc) && return DataFrames.DataFrame()
    return out_acc
end

@inline _spatial_lookup(spatial_info::AbstractDict{Symbol}, key::Symbol, default) = get(spatial_info, key, default)
@inline _spatial_lookup(spatial_info, key::Symbol, default) = hasproperty(spatial_info, key) ? getproperty(spatial_info, key) : default

@inline _spatial_required(spatial_info::AbstractDict{Symbol}, key::Symbol) = spatial_info[key]
@inline _spatial_required(spatial_info, key::Symbol) = getproperty(spatial_info, key)

@inline _field_get(fields::AbstractDict, key::Symbol) = fields[key]
@inline _field_get(fields::NamedTuple, key::Symbol) = getproperty(fields, key)

end # module
