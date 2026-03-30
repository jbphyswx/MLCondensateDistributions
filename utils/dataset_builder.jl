module DatasetBuilder

using DataFrames: DataFrames
using ..CoarseGraining: CoarseGraining
using ..Dynamics: Dynamics

export process_abstract_chunk, SCHEMA_SYMBOL_ORDER

"""
    SCHEMA_SYMBOL_ORDER

The strictly enforced, 31-column canonical sequence for the ML training dataset.
Modifying this tuple requires a collective update to the ML dataloaders and `dataset_spec.md`.
"""
const SCHEMA_SYMBOL_ORDER = (
    :qt, :theta_li, :p, :rho, :w, :q_liq, :q_ice, :q_con, :tke,
    :var_qt, :var_ql, :var_qi, :var_w, :var_h,
    :cov_qt_ql, :cov_qt_qi, :cov_qt_w, :cov_qt_h,
    :cov_ql_qi, :cov_ql_w, :cov_ql_h, :cov_qi_w, :cov_qi_h, :cov_w_h,
    :resolution_h, :resolution_z, :data_source, :month, :cfSite_number, :forcing_model, :experiment
)

"""
    flatten_and_filter!(df::DataFrames.DataFrame, fields_3d::Dict{Symbol, <:AbstractArray}, fields_1d::Dict{Symbol, <:AbstractArray}, fields_0d::Dict{Symbol, Any}, mask::BitArray{3})

Flattens the coarse-grained 3D blocks into arrays, instantly dropping any cell where `mask` is logically true.
Constructs the `DataFrame` securely in column-major order for massive memory optimization.
Uses categorical lookup (3D fields, 1D vertical profiles, 0D scalars) to ensure data provenance and avoid 3D allocations for resolution fields.
"""
function flatten_and_filter!(df::DataFrames.DataFrame, fields_3d::Dict{Symbol, <:AbstractArray}, fields_1d::Dict{Symbol, <:AbstractArray}, fields_0d::Dict{Symbol, Any}, mask::BitArray{3})
    # Identify exactly which multi-dimensional CartesianIndices survive the Sparsity Mask
    valid_indices = findall(.!mask)
    n_valid = length(valid_indices)
    
    # Return immediately if the block has zero valid cloudy structures
    if n_valid == 0
        return
    end
    
    # Enforce the canonical sequence rigidly from the static module constant
    for k in SCHEMA_SYMBOL_ORDER
        if haskey(fields_0d, k)
            df[!, k] = fill(fields_0d[k], n_valid)
        elseif haskey(fields_1d, k)
            # Handle 1D profiles (e.g. per-level dz) by Extracting the vertical index (k) from CartesianIndex(i,j,k)
            profile = fields_1d[k]
            df[!, k] = [Float32(profile[idx[3]]) for idx in valid_indices]
        elseif haskey(fields_3d, k)
            df[!, k] = Float32.(fields_3d[k][valid_indices])
        else
            # Explicit failure if a required column is missing from all sources
            throw(KeyError("Required schema column $k missing from all data sources (scalars, profiles, and 3D fields). Check pipeline provenance."))
        end
    end
end

"""
    process_abstract_chunk(fine_fields::Dict{String, AbstractArray}, metadata::Dict{Symbol, Any}, spatial_info::Dict{Symbol, Any})

Given a set of pure high-resolution physical arrays (e.g. `ta`, `wa`, `clw`), 
performs Spatial Coarse-Graining, covariance matrix derivation, and returns 
a filtered column-mapped Tabular DataFrame. Now accepts `spatial_info` for 
locally correct grid resolution (dx, dz) extraction.
"""
function process_abstract_chunk(fine_fields::Dict{String, <:AbstractArray}, metadata::Dict{Symbol, Any}, spatial_info::Dict{Symbol, Any})
    # 1. Statically extract REQUIRED physical fields. Any missing elements will natively produce a strict `KeyError`.
    base_qt  = fine_fields["hus"]
    base_h   = fine_fields["thetali"]
    base_p   = fine_fields["pfull"]
    base_rho = fine_fields["rhoa"]
    base_w   = fine_fields["wa"]
    base_ql  = fine_fields["clw"]
    base_qi  = fine_fields["cli"]
    
    # Derived Foundational Fields
    base_q_con = base_ql .+ base_qi
    base_tke   = Dynamics.calc_tke.(fine_fields["ua"], fine_fields["va"], base_w)
    
    # 2. Build Pre-Averaged Multiplication Arrays for the Reynolds decomposition (Covariances)
    prod_qt_qt = base_qt .* base_qt
    prod_ql_ql = base_ql .* base_ql
    prod_qi_qi = base_qi .* base_qi
    prod_w_w   = base_w  .* base_w
    prod_h_h   = base_h  .* base_h
    
    prod_qt_ql = base_qt .* base_ql
    prod_qt_qi = base_qt .* base_qi
    prod_qt_w  = base_qt .* base_w
    prod_qt_h  = base_qt .* base_h
    
    prod_ql_qi = base_ql .* base_qi
    prod_ql_w  = base_ql .* base_w
    prod_ql_h  = base_ql .* base_h
    
    prod_qi_w  = base_qi .* base_w
    prod_qi_h  = base_qi .* base_h
    
    prod_w_h   = base_w  .* base_h
    
    # 3. Perform the rigorous 2x2 Spatial Block Mean Mapping directly over the high-res native structure grids
    cg_qt    = CoarseGraining.cg_2x2_horizontal(base_qt)
    cg_h     = CoarseGraining.cg_2x2_horizontal(base_h)
    cg_p     = CoarseGraining.cg_2x2_horizontal(base_p)
    cg_rho   = CoarseGraining.cg_2x2_horizontal(base_rho)
    cg_w     = CoarseGraining.cg_2x2_horizontal(base_w)
    cg_ql    = CoarseGraining.cg_2x2_horizontal(base_ql)
    cg_qi    = CoarseGraining.cg_2x2_horizontal(base_qi)
    cg_q_con = CoarseGraining.cg_2x2_horizontal(base_q_con)
    cg_tke   = CoarseGraining.cg_2x2_horizontal(base_tke)
    
    # 4. Perform identical spatial collapsing of the cross-multiplications
    cg_prod_qt_qt = CoarseGraining.cg_2x2_horizontal(prod_qt_qt)
    cg_prod_ql_ql = CoarseGraining.cg_2x2_horizontal(prod_ql_ql)
    cg_prod_qi_qi = CoarseGraining.cg_2x2_horizontal(prod_qi_qi)
    cg_prod_w_w   = CoarseGraining.cg_2x2_horizontal(prod_w_w)
    cg_prod_h_h   = CoarseGraining.cg_2x2_horizontal(prod_h_h)
    
    cg_prod_qt_ql = CoarseGraining.cg_2x2_horizontal(prod_qt_ql)
    cg_prod_qt_qi = CoarseGraining.cg_2x2_horizontal(prod_qt_qi)
    cg_prod_qt_w  = CoarseGraining.cg_2x2_horizontal(prod_qt_w)
    cg_prod_qt_h  = CoarseGraining.cg_2x2_horizontal(prod_qt_h)
    
    cg_prod_ql_qi = CoarseGraining.cg_2x2_horizontal(prod_ql_qi)
    cg_prod_ql_w  = CoarseGraining.cg_2x2_horizontal(prod_ql_w)
    cg_prod_ql_h  = CoarseGraining.cg_2x2_horizontal(prod_ql_h)
    
    cg_prod_qi_w  = CoarseGraining.cg_2x2_horizontal(prod_qi_w)
    cg_prod_qi_h  = CoarseGraining.cg_2x2_horizontal(prod_qi_h)
    
    cg_prod_w_h   = CoarseGraining.cg_2x2_horizontal(prod_w_h)
    
    # 5. Formulate Exact Covariances mathematically equivalent to the Subgrid-Scale Variance definition
    fields_3d = Dict{Symbol, AbstractArray{Float32, 3}}(
        # Base Linear Vector Means
        :qt       => cg_qt,
        :theta_li => cg_h,
        :p        => cg_p,
        :rho      => cg_rho,
        :w        => cg_w,
        :q_liq    => cg_ql,
        :q_ice    => cg_qi,
        :q_con    => cg_q_con,
        :tke      => cg_tke,
        
        # Variances
        :var_qt   => CoarseGraining.compute_covariance(cg_prod_qt_qt, cg_qt, cg_qt),
        :var_ql   => CoarseGraining.compute_covariance(cg_prod_ql_ql, cg_ql, cg_ql),
        :var_qi   => CoarseGraining.compute_covariance(cg_prod_qi_qi, cg_qi, cg_qi),
        :var_w    => CoarseGraining.compute_covariance(cg_prod_w_w,   cg_w,  cg_w),
        :var_h    => CoarseGraining.compute_covariance(cg_prod_h_h,   cg_h,  cg_h),
        
        # Covariances
        :cov_qt_ql => CoarseGraining.compute_covariance(cg_prod_qt_ql, cg_qt, cg_ql),
        :cov_qt_qi => CoarseGraining.compute_covariance(cg_prod_qt_qi, cg_qt, cg_qi),
        :cov_qt_w  => CoarseGraining.compute_covariance(cg_prod_qt_w,  cg_qt, cg_w),
        :cov_qt_h  => CoarseGraining.compute_covariance(cg_prod_qt_h,  cg_qt, cg_h),
        
        :cov_ql_qi => CoarseGraining.compute_covariance(cg_prod_ql_qi, cg_ql, cg_qi),
        :cov_ql_w  => CoarseGraining.compute_covariance(cg_prod_ql_w,  cg_ql, cg_w),
        :cov_ql_h  => CoarseGraining.compute_covariance(cg_prod_ql_h,  cg_ql, cg_h),
        
        :cov_qi_w  => CoarseGraining.compute_covariance(cg_prod_qi_w,  cg_qi, cg_w),
        :cov_qi_h  => CoarseGraining.compute_covariance(cg_prod_qi_h,  cg_qi, cg_h),
        
        :cov_w_h   => CoarseGraining.compute_covariance(cg_prod_w_h,   cg_w,  cg_h),
    )
    
    # 6. Categorize 1D Vertical Profiles and 0D Metadata Scalars
    fields_1d = Dict{Symbol, AbstractVector{Float32}}(
        :resolution_z => Float32.(spatial_info[:dz_native_profile])
    )
    
    fields_0d = Dict{Symbol, Any}(
        :resolution_h => Float32(spatial_info[:dx_native] * 2.0)
    )
    # Merge simulation-level metadata (month, cfSite_number, etc.) into the 0D category
    merge!(fields_0d, metadata)

    # 7. Apply physical mask constraint. Atmospheric ML discards mathematically null clouds.
    sparse_mask = cg_q_con .< 1e-8
    
    df = DataFrames.DataFrame()
    flatten_and_filter!(df, fields_3d, fields_1d, fields_0d, sparse_mask)
    
    return df
end

end
