module GoogleLES
    #=
        Remote Data is at https://github.com/google-research/swirl-lm/blob/main/swirl_lm/example/geo_flows/cloud_feedback/README.md   

        Bulk statistics: https://console.cloud.google.com/storage/browser/cloudbench-statistics;tab=objects?prefix=&forceOnObjectsSortingFiltering=false (not what we need but useful to have)

        Full output: https://console.cloud.google.com/storage/browser/cloudbench-simulation-output;tab=objects?prefix=&forceOnObjectsSortingFiltering=false (500 TB, so we need to stream data)


        Potentially https://github.com/JuliaCloud/GoogleCloud.jl is helpful but idk. maybe also roll our own.

        julia> ds = GL.load_zarr_simulation(0, 1, "amip")

        [ Info: Loading Zarr metadata from: https://storage.googleapis.com/cloudbench-simulation-output/0/1/amip/data.zarr
        ZarrGroup at Zarr.GCStore("cloudbench-simulation-output") and path 0/1/amip/data.zarr
        Variables: u T theta_li_diffusive_flux_z sfc_flux_rad_sw theta_li_les_tendency x q_t_les_tendency extended_rad_flux_lw rad_flux_lw_up asr cre_sw
        cloud_fraction p_ref q_t t z sfc_heat_flux_sensible y lwp v cre_lw sfc_heat_flux_latent w rad_heat_src extended_rad_flux_sw q_r
        rho olr q_s q_t_microphysics_source sfc_flux_rad_lw rad_flux_sw q_t_diffusive_flux_z q_c cloud_cover rad_flux_lw theta_li_microphysics_source p theta_li 
        
        # ------------------------------------------------------------------- #
        # KEY VARIABLES METADATA (from Swirl-LM documentation)
        # ------------------------------------------------------------------- #
        # 3D Variables (t, x, y, z):
        #   T              : Air temperature (K)
        #   q_t            : Total water specific humidity (kg/kg)
        #   q_c            : Condensed phase specific humidity (kg/kg)
        #   q_r            : Rain droplet mass fraction (kg/kg)
        #   q_s            : Snow mass fraction (kg/kg)
        #   u, v, w        : Zonal, Meridional, and Vertical velocities (m/s)
        #   p, p_ref       : Hydrodynamic & Reference pressures (Pa)
        #   rho            : Air density (kg/m^3)
        #   theta_li       : Liquid-ice potential temperature (K)
        #   rad_flux_lw/sw : Net longwave/shortwave radiative fluxes (W/m^2)
        # 
        # 1D/2D Variables:
        #   lwp, iwp       : Liquid & Ice water paths (kg/m^2)
        #   cloud_cover    : Column cloud fraction (unitless)
        #   cloud_fraction : Profile cloud fraction (t, z) (unitless)
        #   cre_lw, cre_sw : Longwave & Shortwave Cloud Radiative Effects (W/m^2)
        #   olr, asr       : Outgoing Longwave & Absorbed Shortwave radiation at TOA
        # ------------------------------------------------------------------- #
    =#


using Zarr: Zarr


const GoogleLES_to_CliMAVars_translation = Dict{String, String}(
    "T"        => "ta",         # Air temperature
    "q_t"      => "hus",        # Total water specific humidity
    "u"        => "ua",         # Zonal velocity
    "v"        => "va",         # Meridional velocity
    "w"        => "wa",         # Vertical velocity
    "p_ref"    => "pfull",      # Reference pressure
    "p"        => "pfull",      # Hydrodynamic pressure
    "rho"      => "rhoa",       # Air density
    "theta_li" => "thetali",    # Liquid-ice potential temperature (h)
    "q_r"      => "husra",      # Rain water specific humidity
    "q_s"      => "hussn",      # Snow water specific humidity
)

# ------------------------------------------------------------------- #
# Google LES (Swirl-LM) Specific Physics
# ------------------------------------------------------------------- #
"""
    liquid_frac(T::FT) where {FT}

Compute the liquid fraction of the total condensate `q_c` as a function of temperature `T`.
Follows the Swirl-LM / GoogleLES convention of a linear ramp between T_icenuc and T_freeze.
- For T >= 273.15, fraction is 1.0 (all liquid).
- For T <= 233.0, fraction is 0.0 (all ice).

Uses the provided type `FT` to ensure type stability at whatever precision the data is loaded in.
"""
@inline function liquid_frac(T::FT) where {FT}
    t_icenuc = FT(233.0)
    t_freeze = FT(273.15)
    
    # Clamp the fraction between 0.0 and 1.0
    f_l = (T - t_icenuc) / (t_freeze - t_icenuc)
    return clamp(f_l, FT(0), FT(1))
end

"""
    partition_condensate(q_c::FT, T::FT) where {FT}

Splits total GoogleLES condensate `q_c` into liquid water specific humidity `q_l` 
and ice water specific humidity `q_i` based on the temperature `T`.

Returns: `(q_l, q_i)` of type `FT`
"""
@inline function partition_condensate(q_c::FT, T::FT) where {FT}
    f_l = liquid_frac(T)
    q_l = q_c * f_l
    q_i = q_c * (FT(1) - f_l)
    return q_l, q_i
end

# Base URLs for public GCS buckets
const RAW_OUTPUT_BUCKET = "https://storage.googleapis.com/cloudbench-simulation-output"
const STATS_BUCKET = "https://storage.googleapis.com/cloudbench-statistics"

"""
    load_zarr_simulation(site_id::Int, month::Int, experiment::String="amip")

Lazily load the 3D Zarr data from Google Cloud Storage over HTTPS.
Because the raw data is massive (500 TB), Zarr.jl will only load the chunk metadata initially.

# Arguments
- `site_id`: Int (0 to 499)
- `month`: Int (1, 4, 7, 10)
- `experiment`: String (e.g. "amip", "amip-p4k")
"""
function load_zarr_simulation(site_id::Int, month::Int, experiment::String="amip")
    # Build the URL: https://storage.googleapis.com/cloudbench-simulation-output/0/1/amip/data.zarr
    path = join(
        [RAW_OUTPUT_BUCKET, string(site_id), string(month), experiment, "data.zarr"],
        "/"
    )
    @info "Loading Zarr metadata from: $path"
    
    # zopen handles HTTP URLs transparently
    try
        z = Zarr.zopen(path)
        return z
    catch e
        @error "Failed to open Zarr store at $path" exception=(e, catch_backtrace())
        return nothing
    end
end

"""
    build_tabular(args...; kwargs...)

Function stub for the tabular building orchestrator. 
Implementation is provided in `utils/build_training_data.jl`.
"""
function build_tabular end

end # module