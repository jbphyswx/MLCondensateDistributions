module DatasetBuilder

using DataFrames: DataFrames
using ..CoarseGraining: CoarseGraining
using ..Dynamics: Dynamics

export process_abstract_chunk, SCHEMA_SYMBOL_ORDER

const CLOUD_PRESENCE_THRESHOLD = 1f-10

"""
    SCHEMA_SYMBOL_ORDER

The strictly enforced, 36-column canonical sequence for the ML training dataset.
"""
const SCHEMA_SYMBOL_ORDER = (
    :qt, :theta_li, :ta, :p, :rho, :w, :q_liq, :q_ice, :q_con,
    :liq_fraction, :ice_fraction, :cloud_fraction,
    :tke,
    :var_qt, :var_ql, :var_qi, :var_w, :var_h,
    :cov_qt_ql, :cov_qt_qi, :cov_qt_w, :cov_qt_h,
    :cov_ql_qi, :cov_ql_w, :cov_ql_h, :cov_qi_w, :cov_qi_h, :cov_w_h,
    :resolution_h, :domain_h, :resolution_z, :data_source, :month, :cfSite_number, :forcing_model, :experiment,
)

@inline function _indicator_01(x::Real, threshold::Float32)
    return x > threshold ? 1f0 : 0f0
end

"""
    flatten_and_filter!(df::DataFrames.DataFrame, mask::AbstractArray{Bool,3}, ...)

Flatten coarse-grained 3D arrays into schema-ordered DataFrame columns,
dropping cells where `mask` is true. Uses one pass over `(i,j,k)` (column-major order)
with preallocated columns — no `findall` / per-column re-scans.
"""
function flatten_and_filter!(
    df::DataFrames.DataFrame,
    mask::AbstractArray{Bool,3},
    qt::AbstractArray{FT, 3},
    theta_li::AbstractArray{FT, 3},
    ta::AbstractArray{FT, 3},
    p::AbstractArray{FT, 3},
    rho::AbstractArray{FT, 3},
    w::AbstractArray{FT, 3},
    q_liq::AbstractArray{FT, 3},
    q_ice::AbstractArray{FT, 3},
    liq_fraction::AbstractArray{FT, 3},
    ice_fraction::AbstractArray{FT, 3},
    cloud_fraction::AbstractArray{FT, 3},
    tke::AbstractArray{FT, 3},
    var_qt::AbstractArray{FT, 3},
    var_ql::AbstractArray{FT, 3},
    var_qi::AbstractArray{FT, 3},
    var_w::AbstractArray{FT, 3},
    var_h::AbstractArray{FT, 3},
    cov_qt_ql::AbstractArray{FT, 3},
    cov_qt_qi::AbstractArray{FT, 3},
    cov_qt_w::AbstractArray{FT, 3},
    cov_qt_h::AbstractArray{FT, 3},
    cov_ql_qi::AbstractArray{FT, 3},
    cov_ql_w::AbstractArray{FT, 3},
    cov_ql_h::AbstractArray{FT, 3},
    cov_qi_w::AbstractArray{FT, 3},
    cov_qi_h::AbstractArray{FT, 3},
    cov_w_h::AbstractArray{FT, 3},
    dz_profile::AbstractVector{FT},
    resolution_h::FT,
    domain_h::FT,
    metadata,
) where  {FT <: Real}

    nx, ny, nz = size(mask)
    size(qt) == (nx, ny, nz) || throw(DimensionMismatch("qt shape $(size(qt)) vs mask $(size(mask))"))

    n_valid = 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        n_valid += !mask[i, j, k]
    end
    if n_valid == 0
        return
    end

    col_qt = Vector{Float32}(undef, n_valid)
    col_theta_li = Vector{Float32}(undef, n_valid)
    col_ta = Vector{Float32}(undef, n_valid)
    col_p = Vector{Float32}(undef, n_valid)
    col_rho = Vector{Float32}(undef, n_valid)
    col_w = Vector{Float32}(undef, n_valid)
    col_q_liq = Vector{Float32}(undef, n_valid)
    col_q_ice = Vector{Float32}(undef, n_valid)
    col_q_con = Vector{Float32}(undef, n_valid)
    col_liq_fraction = Vector{Float32}(undef, n_valid)
    col_ice_fraction = Vector{Float32}(undef, n_valid)
    col_cloud_fraction = Vector{Float32}(undef, n_valid)
    col_tke = Vector{Float32}(undef, n_valid)
    col_var_qt = Vector{Float32}(undef, n_valid)
    col_var_ql = Vector{Float32}(undef, n_valid)
    col_var_qi = Vector{Float32}(undef, n_valid)
    col_var_w = Vector{Float32}(undef, n_valid)
    col_var_h = Vector{Float32}(undef, n_valid)
    col_cov_qt_ql = Vector{Float32}(undef, n_valid)
    col_cov_qt_qi = Vector{Float32}(undef, n_valid)
    col_cov_qt_w = Vector{Float32}(undef, n_valid)
    col_cov_qt_h = Vector{Float32}(undef, n_valid)
    col_cov_ql_qi = Vector{Float32}(undef, n_valid)
    col_cov_ql_w = Vector{Float32}(undef, n_valid)
    col_cov_ql_h = Vector{Float32}(undef, n_valid)
    col_cov_qi_w = Vector{Float32}(undef, n_valid)
    col_cov_qi_h = Vector{Float32}(undef, n_valid)
    col_cov_w_h = Vector{Float32}(undef, n_valid)
    col_resolution_z = Vector{Float32}(undef, n_valid)

    pos = 1
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        mask[i, j, k] && continue
        col_qt[pos] = Float32(qt[i, j, k])
        col_theta_li[pos] = Float32(theta_li[i, j, k])
        col_ta[pos] = Float32(ta[i, j, k])
        col_p[pos] = Float32(p[i, j, k])
        col_rho[pos] = Float32(rho[i, j, k])
        col_w[pos] = Float32(w[i, j, k])
        ql_ij = Float32(q_liq[i, j, k])
        qi_ij = Float32(q_ice[i, j, k])
        col_q_liq[pos] = ql_ij
        col_q_ice[pos] = qi_ij
        col_q_con[pos] = ql_ij + qi_ij
        col_liq_fraction[pos] = Float32(liq_fraction[i, j, k])
        col_ice_fraction[pos] = Float32(ice_fraction[i, j, k])
        col_cloud_fraction[pos] = Float32(cloud_fraction[i, j, k])
        col_tke[pos] = Float32(tke[i, j, k])
        col_var_qt[pos] = Float32(var_qt[i, j, k])
        col_var_ql[pos] = Float32(var_ql[i, j, k])
        col_var_qi[pos] = Float32(var_qi[i, j, k])
        col_var_w[pos] = Float32(var_w[i, j, k])
        col_var_h[pos] = Float32(var_h[i, j, k])
        col_cov_qt_ql[pos] = Float32(cov_qt_ql[i, j, k])
        col_cov_qt_qi[pos] = Float32(cov_qt_qi[i, j, k])
        col_cov_qt_w[pos] = Float32(cov_qt_w[i, j, k])
        col_cov_qt_h[pos] = Float32(cov_qt_h[i, j, k])
        col_cov_ql_qi[pos] = Float32(cov_ql_qi[i, j, k])
        col_cov_ql_w[pos] = Float32(cov_ql_w[i, j, k])
        col_cov_ql_h[pos] = Float32(cov_ql_h[i, j, k])
        col_cov_qi_w[pos] = Float32(cov_qi_w[i, j, k])
        col_cov_qi_h[pos] = Float32(cov_qi_h[i, j, k])
        col_cov_w_h[pos] = Float32(cov_w_h[i, j, k])
        col_resolution_z[pos] = Float32(dz_profile[k])
        pos += 1
    end

    meta_ds = _metadata_lookup(metadata, :data_source, "unknown")
    meta_mo = _metadata_lookup(metadata, :month, -1)
    meta_cf = _metadata_lookup(metadata, :cfSite_number, -1)
    meta_fm = _metadata_lookup(metadata, :forcing_model, "unknown")
    meta_ex = _metadata_lookup(metadata, :experiment, "unknown")

    df[!, :qt] = col_qt
    df[!, :theta_li] = col_theta_li
    df[!, :ta] = col_ta
    df[!, :p] = col_p
    df[!, :rho] = col_rho
    df[!, :w] = col_w
    df[!, :q_liq] = col_q_liq
    df[!, :q_ice] = col_q_ice
    df[!, :q_con] = col_q_con
    df[!, :liq_fraction] = col_liq_fraction
    df[!, :ice_fraction] = col_ice_fraction
    df[!, :cloud_fraction] = col_cloud_fraction
    df[!, :tke] = col_tke
    df[!, :var_qt] = col_var_qt
    df[!, :var_ql] = col_var_ql
    df[!, :var_qi] = col_var_qi
    df[!, :var_w] = col_var_w
    df[!, :var_h] = col_var_h
    df[!, :cov_qt_ql] = col_cov_qt_ql
    df[!, :cov_qt_qi] = col_cov_qt_qi
    df[!, :cov_qt_w] = col_cov_qt_w
    df[!, :cov_qt_h] = col_cov_qt_h
    df[!, :cov_ql_qi] = col_cov_ql_qi
    df[!, :cov_ql_w] = col_cov_ql_w
    df[!, :cov_ql_h] = col_cov_ql_h
    df[!, :cov_qi_w] = col_cov_qi_w
    df[!, :cov_qi_h] = col_cov_qi_h
    df[!, :cov_w_h] = col_cov_w_h
    df[!, :resolution_h] = fill(resolution_h, n_valid)
    df[!, :domain_h] = fill(domain_h, n_valid)
    df[!, :resolution_z] = col_resolution_z
    df[!, :data_source] = fill(meta_ds, n_valid)
    df[!, :month] = fill(meta_mo, n_valid)
    df[!, :cfSite_number] = fill(meta_cf, n_valid)
    df[!, :forcing_model] = fill(meta_fm, n_valid)
    df[!, :experiment] = fill(meta_ex, n_valid)
    return nothing
end

@inline _metadata_lookup(metadata::AbstractDict{Symbol}, key::Symbol, default) = get(metadata, key, default)
@inline _metadata_lookup(metadata, key::Symbol, default) = hasproperty(metadata, key) ? getproperty(metadata, key) : default

include("dataset_builder_impl.jl")

"""
    process_abstract_chunk(fine_fields, metadata, spatial_info)

Default dataset-builder entrypoint.
This routes to the current implementation in `DatasetBuilderImpl`.
"""
@inline function process_abstract_chunk(fine_fields, metadata, spatial_info)
    return DatasetBuilderImpl.process_abstract_chunk_impl(fine_fields, metadata, spatial_info)
end

end # module DatasetBuilder
