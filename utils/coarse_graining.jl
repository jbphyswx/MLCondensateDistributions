module CoarseGraining

using Statistics: Statistics

export cg_2x2_horizontal, cg_2x_vertical, cg_horizontal_factor, cg_horizontal_product_factor, compute_covariance,
    compute_z_coarsening_scheme, identify_empty_z_levels, identify_empty_z_levels_from_ql_qi, build_z_level_keep_mask, apply_z_level_mask_to_field, build_z_profile_after_mask

"""
    cg_2x2_horizontal(field::AbstractArray{T, 3}) where T

Reduces a 3D field horizontally by evaluating the exact mean over non-overlapping `2x2` blocks.
Drops the final row or column if the horizontal dimensions are odd.
Returns a new array of size `(nx ÷ 2, ny ÷ 2, nz)`.
"""
function cg_2x2_horizontal(field::AbstractArray{T, 3}) where T
    nx, ny, nz = size(field)
    nxc = div(nx, 2)
    nyc = div(ny, 2)
    
    coarse = similar(field, nxc, nyc, nz)
    @inbounds for k in 1:nz
        for j in 1:nyc
            fj = 2j - 1
            for i in 1:nxc
                fi = 2i - 1
                coarse[i, j, k] = T(0.25) * (
                    field[fi,   fj,   k] + 
                    field[fi+1, fj,   k] + 
                    field[fi,   fj+1, k] + 
                    field[fi+1, fj+1, k]
                )
            end
        end
    end
    return coarse
end

"""
    cg_horizontal_factor(field::AbstractArray{T, 3}, factor::Int) where T

Reduces a 3D field horizontally by an integer `factor` (power-of-two expected in this workflow)
by averaging non-overlapping `factor x factor` blocks.
"""
function cg_horizontal_factor(field::AbstractArray{T, 3}, factor::Int) where T
    factor >= 1 || throw(ArgumentError("factor must be >= 1"))
    if factor == 1
        return field
    end

    nx, ny, nz = size(field)
    nxc = div(nx, factor)
    nyc = div(ny, factor)
    inv_area = T(1) / T(factor * factor)

    coarse = similar(field, nxc, nyc, nz)
    @inbounds for k in 1:nz
        for j in 1:nyc
            base_j = (j - 1) * factor + 1
            for i in 1:nxc
                base_i = (i - 1) * factor + 1
                acc = zero(T)
                for fj in 0:(factor - 1)
                    jj = base_j + fj
                    for fi in 0:(factor - 1)
                        acc += field[base_i + fi, jj, k]
                    end
                end
                coarse[i, j, k] = acc * inv_area
            end
        end
    end
    return coarse
end

"""
    cg_horizontal_product_factor(x::AbstractArray{Tx, 3}, y::AbstractArray{Ty, 3}, factor::Int)

Directly computes coarse-grained `<x*y>` over horizontal `factor x factor` blocks without
materializing a full-resolution product array.
"""
function cg_horizontal_product_factor(x::AbstractArray{Tx, 3}, y::AbstractArray{Ty, 3}, factor::Int) where {Tx, Ty}
    factor >= 1 || throw(ArgumentError("factor must be >= 1"))
    size(x) == size(y) || throw(ArgumentError("x and y must have matching dimensions"))

    nx, ny, nz = size(x)
    nxc = div(nx, factor)
    nyc = div(ny, factor)
    T = promote_type(Tx, Ty)
    inv_area = T(1) / T(factor * factor)

    coarse = similar(x, T, nxc, nyc, nz)
    @inbounds for k in 1:nz
        for j in 1:nyc
            base_j = (j - 1) * factor + 1
            for i in 1:nxc
                base_i = (i - 1) * factor + 1
                acc = zero(T)
                for fj in 0:(factor - 1)
                    jj = base_j + fj
                    for fi in 0:(factor - 1)
                        ii = base_i + fi
                        acc += x[ii, jj, k] * y[ii, jj, k]
                    end
                end
                coarse[i, j, k] = acc * inv_area
            end
        end
    end
    return coarse
end

"""
    cg_2x_vertical(field::AbstractArray{T, 3}) where T

Reduces a 3D field vertically by evaluating the exact mean over non-overlapping `2x` blocks.
Drops the uppermost level if the vertical dimension is odd.
Returns a new array of size `(nx, ny, nz ÷ 2)`.
"""
function cg_2x_vertical(field::AbstractArray{T, 3}) where T
    nx, ny, nz = size(field)
    nzc = div(nz, 2)
    
    coarse = similar(field, nx, ny, nzc)
    @inbounds for k in 1:nzc
        fk = 2k - 1
        for j in 1:ny
            for i in 1:nx
                coarse[i, j, k] = T(0.5) * (
                    field[i, j, fk] + 
                    field[i, j, fk+1]
                )
            end
        end
    end
    return coarse
end

"""
    compute_covariance(mean_xy::AbstractArray{T, 3}, mean_x::AbstractArray{T, 3}, mean_y::AbstractArray{T, 3}) where T

Computes the spatially coarse-grained covariance given the block-averaged product field `mean_xy` 
and the independently averaged fields `mean_x` and `mean_y` across the same parent spatial boundaries.
Covariance `Cov(x,y) = <xy> - <x><y>`.
"""
function compute_covariance(mean_xy::AbstractArray{T, 3}, mean_x::AbstractArray{T, 3}, mean_y::AbstractArray{T, 3}) where T
    @assert size(mean_xy) == size(mean_x) == size(mean_y) "All moments must be mapped to the same dimension."
    
    nx, ny, nz = size(mean_x)
    cov = similar(mean_x, nx, ny, nz)
    
    @inbounds for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                # Exact residual mapping inside the reduced block: <x'y'> = <xy> - <x><y>
                cov[i, j, k] = mean_xy[i, j, k] - (mean_x[i, j, k] * mean_y[i, j, k])
            end
        end
    end
    return cov
end

# ===== Vertical Coarsening Infrastructure (Phase 1) =====

"""
    compute_z_coarsening_scheme(dz_profile::AbstractVector{<:Real}, max_dz_target::Float32=400f0)

Builds a vector of vertical coarsening schemes (z_factor, effective_dz) starting from
the native grid up to `max_dz_target` (default 400 m). Each step is binary (2x) coarsening.

Returns a vector of `(z_factor_cumulative, effective_dz)` tuples, where:
- `z_factor_cumulative` is the total 2x reductions applied (e.g., 1, 2, 4, 8, ...)
- `effective_dz` is the resulting effective resolution (sum of averaged dz_profile entries)

**Example**: native dz = [10, 10, 10, ...] (127 levels)
- Returns [(1, 10), (2, 20), (4, 40), (8, 80), (16, 160), (32, 320)]
- Stops before (64, 640) because 640 > 400
"""
function compute_z_coarsening_scheme(dz_profile::AbstractVector{<:Real}, max_dz_target::Float32=400f0)
    nz = length(dz_profile)
    if nz < 1
        error("dz_profile must have at least one level")
    end

    schemes = Tuple{Int, Float32}[]
    
    # Start with native resolution
    native_dz = Statistics.mean(Float32.(dz_profile))
    push!(schemes, (1, native_dz))
    
    # Progressively coarsen by factors of 2
    z_factor = 2
    while nz ÷ z_factor >= 1
        # Estimate effective dz by averaging pairs successively
        effective_dz = native_dz * Float32(z_factor)
        
        if effective_dz > max_dz_target
            break  # Stop coarsening
        end
        
        push!(schemes, (z_factor, effective_dz))
        z_factor *= 2
    end
    
    return schemes
end

"""
    identify_empty_z_levels(q_c_3d::AbstractArray{<:Real, 3}, threshold::Float32=1f-10)

Identifies z-levels where the entire (x,y) plane has `max(q_c) <= threshold`.
Returns a BitVector where `true` means empty (safe to drop at current resolution).

**Safety**: This identifies *potential* drops at the current resolution. The caller
must verify drops won't corrupt future coarsenings (see `build_z_level_keep_mask`).
"""
function identify_empty_z_levels(q_c_3d::AbstractArray{<:Real, 3}, threshold::Float32=1f-10)
    nx, ny, nz = size(q_c_3d)
    empty_mask = BitVector(undef, nz)
    
    @inbounds for k in 1:nz
        max_qc = zero(Float32)
        for j in 1:ny
            for i in 1:nx
                max_qc = max(max_qc, Float32(q_c_3d[i, j, k]))
            end
        end
        empty_mask[k] = max_qc <= threshold
    end
    
    return empty_mask
end

"""
    identify_empty_z_levels_from_ql_qi(ql, qi, threshold)

Same semantics as [`identify_empty_z_levels`](@ref) on `ql .+ qi`, but **without** allocating
the sum array (uses `max(ql+qi)` over each `(x,y)` plane per `k`).
"""
function identify_empty_z_levels_from_ql_qi(
    ql::AbstractArray{<:Real, 3},
    qi::AbstractArray{<:Real, 3},
    threshold::Float32 = 1f-10,
)
    nx, ny, nz = size(ql)
    size(qi) == (nx, ny, nz) || throw(DimensionMismatch("ql and qi must match shape"))
    empty_mask = BitVector(undef, nz)
    @inbounds for k in 1:nz
        max_q = zero(Float32)
        for j in 1:ny
            for i in 1:nx
                max_q = max(max_q, Float32(ql[i, j, k]) + Float32(qi[i, j, k]))
            end
        end
        empty_mask[k] = max_q <= threshold
    end
    return empty_mask
end

"""
    build_z_level_keep_mask(empty_z_levels::BitVector, z_factor::Int, future_z_factors::AbstractVector{Int})

Builds a conservative BitVector indicating which z-levels to keep, ensuring that
no dropped data would be used in future coarsenings.

**Logic**: 
- At z_factor=1 (native), drop only truly isolated empty levels.
- At coarser z_factors, be more conservative to preserve history for finer resolutions.
- For future z_factors (if any), ensure we don't drop an even index if its partner is non-empty.

**Critical**: Failure to respect this can cause statistical skew. Use extensive testing.
"""
function build_z_level_keep_mask(
    empty_z_levels::BitVector,
    z_factor::Int,
    future_z_factors::AbstractVector{Int},
)
    nz = length(empty_z_levels)
    
    # Conservative strategy: for each potential pair/group, only drop if:
    # - All members of the group are empty at this resolution
    # - AND dropping won't affect future coarsenings
    
    # If this is the coarsest resolution we care about, we can be aggressive
    is_coarsest = isempty(future_z_factors) || Statistics.maximum(future_z_factors) <= z_factor
    
    keep_mask = trues(nz)
    
    if is_coarsest
        # At coarsest, simply copy the empty_z_levels complement
        keep_mask = .!empty_z_levels
    else
        # At finer resolutions, be conservative: only drop isolated runs of empties
        # that won't affect future coarsenings. 
        # For now, a simple heuristic: keep edge levels (bottom & top) and 
        # isolated pockets; drop only if surrounded by condensate or at boundaries.
        
        for k in 1:nz
            if empty_z_levels[k]
                # Check if this empty level is surrounded or isolated
                has_neighbor_with_cloud = false
                
                # Check neighbors within current group
                for offset in 1:z_factor
                    if k + offset <= nz && !empty_z_levels[k + offset]
                        has_neighbor_with_cloud = true
                        break
                    end
                    if k - offset >= 1 && !empty_z_levels[k - offset]
                        has_neighbor_with_cloud = true
                        break
                    end
                end
                
                # Conservative: only drop if surrounded OR if at an interior point
                # Preserve boundary and transitional levels
                if !has_neighbor_with_cloud && k > 1 && k < nz
                    keep_mask[k] = false  # Safe to drop isolated interior empty
                end
            end
        end
    end
    
    return keep_mask
end

"""
    apply_z_level_mask_to_field(field_3d::AbstractArray{T, 3}, z_keep_mask::BitVector)

Filters field_3d along the z-axis, keeping only levels where z_keep_mask is true.
Returns a new array with fewer z-levels.
"""
function apply_z_level_mask_to_field(field_3d::AbstractArray{T, 3}, z_keep_mask::BitVector) where T
    nx, ny, nz = size(field_3d)
    length(z_keep_mask) == nz || error("z_keep_mask length $(length(z_keep_mask)) != nz=$nz")
    
    nz_kept = count(z_keep_mask)
    if nz_kept == nz
        return field_3d  # No filtering needed
    end
    
    field_filtered = similar(field_3d, nx, ny, nz_kept)
    
    kept_idx = 0
    @inbounds for k in 1:nz
        if z_keep_mask[k]
            kept_idx += 1
            for j in 1:ny
                for i in 1:nx
                    field_filtered[i, j, kept_idx] = field_3d[i, j, k]
                end
            end
        end
    end
    
    return field_filtered
end

"""
    build_z_profile_after_mask(dz_profile::AbstractVector{<:Real}, z_keep_mask::BitVector)

Updates dz_profile after applying z_level_mask, keeping only the dz values for kept levels.
"""
function build_z_profile_after_mask(dz_profile::AbstractVector{<:Real}, z_keep_mask::BitVector)
    length(z_keep_mask) == length(dz_profile) || error("z_keep_mask and dz_profile length mismatch")
    
    dz_kept = Float32[Float32(dz_profile[k]) for k in 1:length(dz_profile) if z_keep_mask[k]]
    return dz_kept
end

end
