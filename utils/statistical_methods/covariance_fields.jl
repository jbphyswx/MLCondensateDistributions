"""
    covariance_from_moments!(out, mean_xy, mean_x, mean_y)

In-place per-voxel covariance from coarse **means** only:

``\\mathrm{Cov}(X,Y) = \\mathbb{E}[XY] - \\mathbb{E}[X]\\mathbb{E}[Y]``,

i.e. `mean_xy .- mean_x .* mean_y` when `mean_xy` is the voxel mean of the product and `mean_x`,
`mean_y` are marginal means (all same shape). For a variance diagonal, pass `mean_x` twice:
`covariance_from_moments!(out, mean_xx, mean_x, mean_x)`.
"""
@inline function covariance_from_moments!(
    out::AbstractArray{T, 3},
    mean_xy::AbstractArray{T, 3},
    mean_x::AbstractArray{T, 3},
    mean_y::AbstractArray{T, 3},
) where {T <: Real}
    @. out = mean_xy - mean_x * mean_y
    return out
end

"""
    covariance_from_moments(mean_xy, mean_x, mean_y)

Allocating version of [`covariance_from_moments!`](@ref).
"""
function covariance_from_moments(
    mean_xy::AbstractArray{T, 3},
    mean_x::AbstractArray{T, 3},
    mean_y::AbstractArray{T, 3},
) where {T <: Real}
    size(mean_xy) == size(mean_x) == size(mean_y) || throw(DimensionMismatch("moment arrays must share shape"))
    return @. mean_xy - mean_x * mean_y
end
