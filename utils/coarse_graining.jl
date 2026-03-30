module CoarseGraining

export cg_2x2_horizontal, cg_2x_vertical, compute_covariance

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

end
