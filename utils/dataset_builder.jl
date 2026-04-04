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

@inline function _gather_3d(arr::AbstractArray{<:Real, 3}, valid_indices::Vector{CartesianIndex{3}})
    n = length(valid_indices)
    out = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        out[i] = Float32(arr[valid_indices[i]])
    end
    return out
end

@inline function _gather_3d_sum(a::AbstractArray{<:Real, 3}, b::AbstractArray{<:Real, 3}, valid_indices::Vector{CartesianIndex{3}})
    n = length(valid_indices)
    out = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        idx = valid_indices[i]
        out[i] = Float32(a[idx] + b[idx])
    end
    return out
end

@inline function _gather_profile(profile::AbstractVector{<:Real}, valid_indices::Vector{CartesianIndex{3}})
    n = length(valid_indices)
    out = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        out[i] = Float32(profile[valid_indices[i][3]])
    end
    return out
end

"""
    flatten_and_filter!(df::DataFrames.DataFrame, mask::BitArray{3}, ...)

Flatten coarse-grained 3D arrays into schema-ordered DataFrame columns,
dropping cells where `mask` is true.
"""
function flatten_and_filter!(
    df::DataFrames.DataFrame,
    mask::BitArray{3},
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

    valid_indices = findall(!, mask)
    n_valid = length(valid_indices)
    if n_valid == 0
        return
    end

    df[!, :qt] = _gather_3d(qt, valid_indices)
    df[!, :theta_li] = _gather_3d(theta_li, valid_indices)
    df[!, :ta] = _gather_3d(ta, valid_indices)
    df[!, :p] = _gather_3d(p, valid_indices)
    df[!, :rho] = _gather_3d(rho, valid_indices)
    df[!, :w] = _gather_3d(w, valid_indices)
    df[!, :q_liq] = _gather_3d(q_liq, valid_indices)
    df[!, :q_ice] = _gather_3d(q_ice, valid_indices)
    df[!, :q_con] = _gather_3d_sum(q_liq, q_ice, valid_indices)
    df[!, :liq_fraction] = _gather_3d(liq_fraction, valid_indices)
    df[!, :ice_fraction] = _gather_3d(ice_fraction, valid_indices)
    df[!, :cloud_fraction] = _gather_3d(cloud_fraction, valid_indices)
    df[!, :tke] = _gather_3d(tke, valid_indices)

    df[!, :var_qt] = _gather_3d(var_qt, valid_indices)
    df[!, :var_ql] = _gather_3d(var_ql, valid_indices)
    df[!, :var_qi] = _gather_3d(var_qi, valid_indices)
    df[!, :var_w] = _gather_3d(var_w, valid_indices)
    df[!, :var_h] = _gather_3d(var_h, valid_indices)

    df[!, :cov_qt_ql] = _gather_3d(cov_qt_ql, valid_indices)
    df[!, :cov_qt_qi] = _gather_3d(cov_qt_qi, valid_indices)
    df[!, :cov_qt_w] = _gather_3d(cov_qt_w, valid_indices)
    df[!, :cov_qt_h] = _gather_3d(cov_qt_h, valid_indices)
    df[!, :cov_ql_qi] = _gather_3d(cov_ql_qi, valid_indices)
    df[!, :cov_ql_w] = _gather_3d(cov_ql_w, valid_indices)
    df[!, :cov_ql_h] = _gather_3d(cov_ql_h, valid_indices)
    df[!, :cov_qi_w] = _gather_3d(cov_qi_w, valid_indices)
    df[!, :cov_qi_h] = _gather_3d(cov_qi_h, valid_indices)
    df[!, :cov_w_h] = _gather_3d(cov_w_h, valid_indices)

    df[!, :resolution_h] = fill(resolution_h, n_valid)
    df[!, :domain_h] = fill(domain_h, n_valid)
    df[!, :resolution_z] = _gather_profile(dz_profile, valid_indices)

    df[!, :data_source] = fill(_metadata_lookup(metadata, :data_source, "unknown"), n_valid)
    df[!, :month] = fill(_metadata_lookup(metadata, :month, -1), n_valid)
    df[!, :cfSite_number] = fill(_metadata_lookup(metadata, :cfSite_number, -1), n_valid)
    df[!, :forcing_model] = fill(_metadata_lookup(metadata, :forcing_model, "unknown"), n_valid)
    df[!, :experiment] = fill(_metadata_lookup(metadata, :experiment, "unknown"), n_valid)
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
