# Consolidated build_tabular logic for all data sources.
# This file centralizes the orchestration of data loading and processing
# into tabular Arrow format.


# utils/build_training_data.jl
# Consolidated Logic Hub for Tabular Data Generation


#=
    Here we build training data to be stored in the training data directory.

    We follow the format laid out in dataset_spec.md

    We can store data in zarr (or tabular, idk which is faster and easier yet for machine learning) format.

    We will go from 1km to native resolution.
    While ideally we could do any size reduction, for speed it should be best for us to do binary size reductions.
    So say we have a 6km x 6km domain, we can do 6km, 3km, 1.5km. The next would be 750m, which is smaller than 1km, so we stop there (but we could do this just by repeated split combines, so split all the way down to 4x4 and then combine to 2x2, then 1x1)

    The vertical is more complicated, since vertical grids may not be uniform (luckily for us it seems they (mostly) are) Let our finest resolution be 10m, and proceed to up to a 4x reduction from the LES.
    So in the example above we'd have 3 horizontal resolutions, and 3 vertical resolutions, so 9 differnt combinations in total.

    Note we have to do some processing to get things like tke from u, v, w, etc.

    This should be a fully parallelizeable and well documented process, that can be run on the HPC as well with slurm, can be distributed, threaded (using OhMyThreas), or just run serially. And can be stopped and resumed, restarted, etc.
    Note many levels, subcolumns etc may have no condensate. For space, we should drop them from the outputs, since those are just null data.
    Note right now, i think the cfSite provides qi and ql, but i think the LES data only provides qc. 

    Right now, GoogleLES is using    
        liquid_frac_nuc = lambda t: (t - self._t_icenuc) / (  # pylint: disable=g-long-lambda 
            self._t_freeze - self._t_icenuc)
        Which we can see by followinga chain through :
            https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/physics/thermodynamics/water.py#L943
            https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/physics/thermodynamics/water.py#L897C2-L931C1
            https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/physics/thermodynamics/thermodynamics.proto#L103
        with values t_icenuc = 233.0 K, t_freeze = 273.15 K from https://github.com/CliMA/CLIMAParameters.jl/blob/main/src/Planet/planet_parameters.jl

    In principle in the future we may want to move to a CiMA type split where we use pow_icenuc as (T - T_icenuc) / (T_freeze - T_icenuc))^pow_icenuc, but for now le'ts just do the simple thing, and recreate the data how it was run.
=#

#=
    Orchestration script for processing Google LES (Swirl-LM) and cfSite high-res datasets
    into ML-ready flattened, coarse-grained tabular matrices.
=#


import .GoogleLES
import .cfSites
import .DatasetBuilder
import Arrow

# ------------------------------------------------------------------- #
# Google LES Implementation
# ------------------------------------------------------------------- #

function GoogleLES.build_tabular(site_id::Int, month::Int, experiment::String, output_dir::String; max_timesteps::Int=0)
    mkpath(output_dir)
    @info "Loading GoogleLES data for site $site_id, month $month, exp $experiment..."
    
    ds = GoogleLES.load_zarr_simulation(site_id, month, experiment)
    if isnothing(ds)
        @error "Could not load simulation."
        return
    end

    # Extract dimensions
    nt = length(ds["t"])
    if max_timesteps > 0
        nt = min(nt, max_timesteps)
    end
    
    x_coords = collect(ds["x"][:])
    z_coords = collect(ds["z"][:])
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    
    # dz varies with z in some grids, compute profile
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end]) 
    
    metadata = Dict{Symbol, Any}(
        :data_source => "GoogleLES",
        :month => month,
        :site_id => site_id,
        :experiment => experiment
    )
    
    spatial_info = Dict{Symbol, Any}(
        :dx_native => Float32(dx_native),
        :dz_native_profile => Float32.(dz_native_profile)
    )

    @info "Processing $nt time-steps..."

    for t_idx in 0:(nt-1)
        out_file = joinpath(output_dir, "googleles_$(experiment)_$(month)_$(site_id)_t$(t_idx).arrow")
        
        if !isfile(out_file)
            @info "Processing timestep $t_idx..."
            fine_fields = Dict{String, AbstractArray{Float32, 3}}()
            
            # Load and translate variables
            for (g_var, c_var) in GoogleLES.GoogleLES_to_CliMAVars_translation
                # GoogleLES Zarr stores [t, x, y, z]
                # We want [x, y, z] at a specific t
                fine_fields[c_var] = Float32.(ds[g_var][t_idx+1, :, :, :])
            end
            
            # Special handling for q_c -> clw, cli
            q_c = Float32.(ds["q_c"][t_idx+1, :, :, :])
            ta = fine_fields["ta"]
            
            clw = similar(q_c)
            cli = similar(q_c)
            
            for i in eachindex(q_c)
                clw[i], cli[i] = GoogleLES.partition_condensate(q_c[i], ta[i])
            end
            
            fine_fields["clw"] = clw
            fine_fields["cli"] = cli
            
            metadata_t = copy(metadata)
            metadata_t[:timestep] = t_idx
            
            df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
            Arrow.write(out_file, df)
        end
    end
end

# ------------------------------------------------------------------- #
# cfSites Implementation
# ------------------------------------------------------------------- #

function cfSites.build_tabular(cfSite_number::Int, month::Int, forcing_model::String, experiment::String, output_dir::String; max_timesteps::Int=0)
    mkpath(output_dir)
    @info "Loading cfSite data for site $cfSite_number, model $forcing_model..."
    
    # Load the 4D field stack
    vars_to_load = ["temperature", "qt", "ql", "qi", "u", "v", "w", "p", "rho", "thetali"]
    les_dir = cfSites.get_cfSite_les_dir(cfSite_number; forcing_model=forcing_model, month=month, experiment=experiment)
    ds_stack = cfSites.load_4d_fields(les_dir, vars_to_load)
    
    # Extract dimensions
    nt = length(DimensionalData.dims(ds_stack, DimensionalData.Ti))
    if max_timesteps > 0
        nt = min(nt, max_timesteps)
    end
    
    x_coords = collect(DimensionalData.dims(ds_stack, DimensionalData.X))
    z_coords = collect(DimensionalData.dims(ds_stack, DimensionalData.Z))
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end])
    
    metadata = Dict{Symbol, Any}(
        :data_source => "cfSites",
        :month => month,
        :cfSite_number => cfSite_number,
        :forcing_model => forcing_model,
        :experiment => experiment
    )
    
    spatial_info = Dict{Symbol, Any}(
        :dx_native => Float32(dx_native),
        :dz_native_profile => Float32.(dz_native_profile)
    )
    
    @info "Processing $nt time-steps..."
    
    for t_idx in 1:nt
        out_file = joinpath(output_dir, "cfsites_$(forcing_model)_$(experiment)_$(month)_$(cfSite_number)_t$(t_idx).arrow")
        
        if !isfile(out_file)
            @info "Processing timestep $t_idx..."
            fine_fields = Dict{String, AbstractArray{Float32, 3}}()
            
            # Map cfSite internal names to the canonical names expected by DatasetBuilder
            translation = Dict(
                "temperature" => "ta",
                "qt"          => "hus",
                "u"           => "ua",
                "v"           => "va",
                "w"           => "wa",
                "p"           => "pfull",
                "rho"         => "rhoa",
                "ql"          => "clw",
                "qi"          => "cli",
                "thetali"     => "thetali",
            )
            
            for (site_var, canonical_var) in translation
                # Slice the 4D DimArray at the current timestep
                fine_fields[canonical_var] = Float32.(ds_stack[Symbol(site_var)][DimensionalData.Ti(t_idx)])
            end
            
            metadata_t = copy(metadata)
            metadata_t[:timestep] = t_idx

            df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
            Arrow.write(out_file, df)
        end
    end
end
