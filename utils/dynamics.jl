module Dynamics

export TKE, KE, variance_from_moments, TKE_from_moments, 
    kinetic_energy, turbulent_kinetic_energy, turbulent_kinetic_energy_from_moments

using Statistics: Statistics



"""
    KE(u, v, w)

Return kinetic energy from raw velocity components.
"""
@inline function KE(u::FT, v::FT, w::FT) where {FT}
    return FT(0.5) * (u^2 + v^2 + w^2)
end

"""
    KE(u, v, w)

Return whole-array kinetic energy from raw velocity samples.
"""
@inline function KE(u::AbstractArray{FT}, v::AbstractArray{FT}, w::AbstractArray{FT}) where {FT}
    return FT(0.5) * (sum(u.^2) + sum(v.^2) + sum(w.^2))
end


# ============================================================================ #

"""
    TKE(u, v, w)

Return turbulent kinetic energy from variances.
The inputs are interpreted as `var(u)`, `var(v)`, and `var(w)`.
"""
@inline function TKE(u_var::FT, v_var::FT, w_var::FT) where {FT}
    return FT(0.5) * (u_var + v_var + w_var)
end


@inline function TKE(u::FT, v::FT, w::FT, u_mean::FT, v_mean::FT, w_mean::FT) where {FT}
   return FT(0.5) * ((u - u_mean)^2 + (v - v_mean)^2 + (w - w_mean)^2)
end

"""
    TKE(u, v, w, u_mean, v_mean, w_mean)

Return whole-array TKE from raw velocity samples and a single mean for each component.
This is a scalar whole-array summary, not a per-cell field.
"""
@inline function TKE(u::AbstractArray{FT}, v::AbstractArray{FT}, w::AbstractArray{FT}, u_mean::FT, v_mean::FT, w_mean::FT) where {FT}
    return FT(0.5) * (sum((u .- u_mean).^2) + sum((v .- v_mean).^2) + sum((w .- w_mean).^2))
end

"""
    TKE(u, v, w)

Return whole-array TKE from raw velocity samples using the global mean of each component.
This is a scalar summary of the arrays.
"""
@inline function TKE(u::AbstractArray{FT}, v::AbstractArray{FT}, w::AbstractArray{FT}; is_variance::Bool=false) where {FT}
    if is_variance
        # Interpret arrays as variance samples and return total TKE across the array.
        return FT(0.5) * (sum(u) + sum(v) + sum(w))
    else
        # Interpret arrays as raw velocity samples and compute fluctuations about the global mean.
        return TKE(u, v, w, Statistics.mean(u), Statistics.mean(v), Statistics.mean(w))
    end
end


# ============================================================================ #



"""
    variance_from_moments(mean_sq, mean)

Compute variance from first and second moments using
`var(x) = <x^2> - <x>^2`.
"""
@inline variance_from_moments(mean_sq::FT, mean::FT) where {FT} = mean_sq - mean * mean

@inline variance_from_moments(mean_sq::AbstractArray{FT}, mean::AbstractArray{FT}) where {FT} = mean_sq .- mean .* mean

"""
    TKE_from_moments(mean_sq_u, mean_u, mean_sq_v, mean_v, mean_sq_w, mean_w)

Compute TKE directly from first and second moments without exposing the
moment-to-variance algebra at the call site.
"""
@inline function TKE_from_moments(
    mean_sq_u::FT, mean_u::FT,
    mean_sq_v::FT, mean_v::FT,
    mean_sq_w::FT, mean_w::FT,
) where {FT}
    var_u = variance_from_moments(mean_sq_u, mean_u)
    var_v = variance_from_moments(mean_sq_v, mean_v)
    var_w = variance_from_moments(mean_sq_w, mean_w)
    return FT(0.5) * (var_u + var_v + var_w)
end

@inline function TKE_from_moments(
    mean_sq_u::AbstractArray{FT}, mean_u::AbstractArray{FT},
    mean_sq_v::AbstractArray{FT}, mean_v::AbstractArray{FT},
    mean_sq_w::AbstractArray{FT}, mean_w::AbstractArray{FT},
) where {FT}
    var_u = variance_from_moments(mean_sq_u, mean_u)
    var_v = variance_from_moments(mean_sq_v, mean_v)
    var_w = variance_from_moments(mean_sq_w, mean_w)
    return FT(0.5) .* (var_u .+ var_v .+ var_w)
end

const kinetic_energy = KE
const turbulent_kinetic_energy = TKE
const turbulent_kinetic_energy_from_moments = TKE_from_moments

end # module
