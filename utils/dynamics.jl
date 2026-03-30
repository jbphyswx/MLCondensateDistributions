module Dynamics

export calc_tke

"""
    calc_tke(u_var::FT, v_var::FT, w_var::FT) where {FT}

Calculate the turbulent kinetic energy (TKE) given the resolved spatial variances 
of the velocity components.
TKE = 0.5 * (u'^2 + v'^2 + w'^2)

Strictly typed to `FT` to prevent type instability when mapped across the LES grids.
"""
@inline function calc_tke(u_var::FT, v_var::FT, w_var::FT) where {FT}
    return FT(0.5) * (u_var + v_var + w_var)
end

end
