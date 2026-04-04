"""
AMIP baseline data generation entrypoint.

This script is designed to be runnable from CLI and from include() workflows.
It writes Arrow files into the package-wide processed-data directory by default.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
using MLCondensateDistributions: MLCondensateDistributions as MLCD

const DEFAULT_SITE_ID = 10
const DEFAULT_MONTH = 1
const DEFAULT_EXPERIMENT = "amip"
const DEFAULT_MAX_TIMESTEPS = 0
const DEFAULT_TIMESTEP_BATCH_SIZE = 0
const DEFAULT_MIN_H_RESOLUTION = 1000.0f0
const DEFAULT_FORCING_MODEL = "HadGEM2-A"
const DEFAULT_VERBOSE_LOGS = false

"""
    generate_data!(; kwargs...)

Generate AMIP baseline training data into Arrow files.

Key behavior:
- GoogleLES generation is enabled by default.
- cfSites generation is optional and gracefully skipped when the required
  HPC fields directory is unavailable on the current machine.
- `max_timesteps=0` means scan the full LES case.
"""
function generate_data!(;
    site_id::Int = DEFAULT_SITE_ID,
    month::Int = DEFAULT_MONTH,
    experiment::String = DEFAULT_EXPERIMENT,
    forcing_model::String = DEFAULT_FORCING_MODEL,
    output_dir::String = MLCD.Paths.processed_data_root(),
    max_timesteps::Int = DEFAULT_MAX_TIMESTEPS,
    timestep_batch_size::Int = DEFAULT_TIMESTEP_BATCH_SIZE,
    min_h_resolution::Float32 = DEFAULT_MIN_H_RESOLUTION,
    include_googleles::Bool = true,
    include_cfsites::Bool = false,
    verbose::Bool = DEFAULT_VERBOSE_LOGS,
    tabular_options::MLCD.TabularBuildOptions = MLCD.TabularBuildOptions(),
)
    mkpath(output_dir)

    println("Generating AMIP baseline data in $(output_dir)")

    if include_googleles
        MLCD.GoogleLES.build_tabular(
            site_id,
            month,
            experiment,
            output_dir;
            max_timesteps=max_timesteps,
            timestep_batch_size=timestep_batch_size,
            min_h_resolution=min_h_resolution,
            verbose=verbose,
            tabular_options=tabular_options,
        )
    end

    if include_cfsites
        les_dir = MLCD.cfSites.get_cfSite_les_dir(site_id; forcing_model=forcing_model, month=month, experiment=experiment)
        fields_dir = joinpath(les_dir, "fields")
        if !isdir(fields_dir)
            if verbose
                @info "Skipping cfSites data generation: fields directory not available on this machine: $(fields_dir)"
            end
        else
            try
                MLCD.cfSites.build_tabular(
                    site_id,
                    month,
                    forcing_model,
                    experiment,
                    output_dir;
                    max_timesteps=max_timesteps,
                    min_h_resolution=min_h_resolution,
                    verbose=verbose,
                    tabular_options=tabular_options,
                )
            catch err
                if verbose
                    @warn "cfSites data generation skipped after read failure: $(sprint(showerror, err))"
                end
            end
        end
    end

    println("Data generation complete: $(output_dir)")
    return output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    include_cfsites = MLCD.EnvHelpers.parse_bool_env("INCLUDE_CFSITES", false)
    max_timesteps = parse(Int, get(ENV, "MAX_TIMESTEPS", string(DEFAULT_MAX_TIMESTEPS)))
    timestep_batch_size = parse(Int, get(ENV, "TIMESTEP_BATCH_SIZE", string(DEFAULT_TIMESTEP_BATCH_SIZE)))
    min_h_resolution = parse(Float32, get(ENV, "MIN_H_RESOLUTION", string(DEFAULT_MIN_H_RESOLUTION)))
    verbose = MLCD.EnvHelpers.parse_bool_env("VERBOSE_GENERATION", DEFAULT_VERBOSE_LOGS)
    tabular_opts = MLCD.tabular_build_options_from_env()
    generate_data!(;
        include_cfsites=include_cfsites,
        max_timesteps=max_timesteps,
        timestep_batch_size=timestep_batch_size,
        min_h_resolution=min_h_resolution,
        verbose=verbose,
        tabular_options=tabular_opts,
    )
end
