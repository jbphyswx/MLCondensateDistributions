"""
Read in cfSite data from Zhaoyi, process it into different resolutions etc.
"""
module cfSites

using NCDatasets: NCDatasets as NC
using DimensionalData: DimensionalData as DD

export get_LES_library
export get_shallow_LES_library
export parse_les_path
export valid_lespath
export get_cfSite_type
export get_cfSite_les_dir
export get_stats_path
export get_les_calibration_library
export get_fields_dir
export get_combined_field_path
export load_4d_fields


const cfSites_to_GoogleLES_translation = Dict{String, String}(
    "temperature" => "T",
    "qt"          => "q_t",
    "u"           => "u",
    "v"           => "v",
    "w"           => "w",
    "p"           => "p",
    "rho"         => "rho",
    "ql"          => "q_l", # NOTE: GoogleLES provides 'q_c', but we split into 'q_l' and 'q_i'
    "qi"          => "q_i",
    "thetali"     => "theta_li",
)

# ========================================================= #
# cfSite Types definitions
const CFSITE_TYPES = Dict(
    "shallow" => vcat(4:15, 17:23),
    "deep" => [30, 31, 32, 33, 66, 67, 68, 69, 70, 82, 92, 94, 96, 99, 100]
)

"""
    get_shallow_LES_library

Hierarchical dictionary of available LES simulations described in Shen et al 2022.
The following cfSites are available across listed models, months,
and experiments.
"""
function get_shallow_LES_library()
    LES_library = Dict(
        "HadGEM2-A" => Dict(),
        "CNRM-CM5" => Dict(),
        "CNRM-CM6-1" => Dict(),
    )
    Shen_et_al_sites = collect(4:15)
    append!(Shen_et_al_sites, collect(17:23))

    # HadGEM2-A model (76 AMIP-AMIP4K pairs)
    LES_library["HadGEM2-A"]["10"] = Dict()
    LES_library["HadGEM2-A"]["10"]["cfSite_numbers"] = Shen_et_al_sites
    LES_library["HadGEM2-A"]["07"] = Dict()
    LES_library["HadGEM2-A"]["07"]["cfSite_numbers"] = deepcopy(Shen_et_al_sites)
    LES_library["HadGEM2-A"]["04"] = Dict()
    LES_library["HadGEM2-A"]["04"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18])
    LES_library["HadGEM2-A"]["01"] = Dict()
    LES_library["HadGEM2-A"]["01"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [15, 17, 18, 19, 20])

    # CNRM-CM5 model (59 AMIP-AMIP4K pairs)
    LES_library["CNRM-CM5"]["10"] = Dict()
    LES_library["CNRM-CM5"]["10"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [15, 22, 23])
    LES_library["CNRM-CM5"]["07"] = Dict()
    LES_library["CNRM-CM5"]["07"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [13, 14, 15, 18])
    LES_library["CNRM-CM5"]["04"] = Dict()
    LES_library["CNRM-CM5"]["04"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [11, 12, 13, 14, 15, 17, 18, 21, 22, 23])
    LES_library["CNRM-CM5"]["01"] = Dict()
    LES_library["CNRM-CM5"]["01"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [14, 15, 17, 18, 19, 20, 21, 22, 23])

    # CNRM-CM6-1 model (69 AMIP-AMIP4K pairs)
    LES_library["CNRM-CM6-1"]["10"] = Dict()
    LES_library["CNRM-CM6-1"]["10"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [22, 23])
    LES_library["CNRM-CM6-1"]["07"] = Dict()
    LES_library["CNRM-CM6-1"]["07"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [12, 13, 14, 15, 17])
    LES_library["CNRM-CM6-1"]["04"] = Dict()
    LES_library["CNRM-CM6-1"]["04"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [13, 14, 15])
    LES_library["CNRM-CM6-1"]["01"] = Dict()
    LES_library["CNRM-CM6-1"]["01"]["cfSite_numbers"] = setdiff(Shen_et_al_sites, [14, 15, 21, 22, 23])

    for month in ["01", "04", "07", "10"]
        LES_library["HadGEM2-A"][month]["experiments"] = ["amip", "amip4K"]
        LES_library["CNRM-CM5"][month]["experiments"] = ["amip", "amip4K"]
        LES_library["CNRM-CM6-1"][month]["experiments"] = ["amip", "amip4K"]
    end
    return LES_library
end

"""
    get_LES_library

Hierarchical dictionary of available cfSite LES simulations.
"""
function get_LES_library()
    LES_library = get_shallow_LES_library()
    deep_sites = (collect(30:33)..., collect(66:70)..., 82, 92, 94, 96, 99, 100)

    append!(LES_library["HadGEM2-A"]["07"]["cfSite_numbers"], deep_sites)
    append!(LES_library["HadGEM2-A"]["01"]["cfSite_numbers"], deep_sites)
    sites_04 = deepcopy(setdiff(deep_sites, [32, 92, 94]))
    append!(LES_library["HadGEM2-A"]["04"]["cfSite_numbers"], sites_04)
    sites_10 = deepcopy(setdiff(deep_sites, [94, 100]))
    append!(LES_library["HadGEM2-A"]["10"]["cfSite_numbers"], sites_10)

    LES_library_full = deepcopy(LES_library)
    for model in keys(LES_library_full)
        for month in keys(LES_library_full[model])
            LES_library_full[model][month]["cfSite_numbers"] = Dict()
            for cfSite_number in LES_library[model][month]["cfSite_numbers"]
                cfSite_number_str = string(cfSite_number, pad = 2)
                LES_library_full[model][month]["cfSite_numbers"][cfSite_number_str] =
                    if cfSite_number >= 30
                        "deep"
                    else
                        "shallow"
                    end
            end
        end
    end
    return LES_library_full
end

"""
    parse_les_path(les_path)
    
Given path to LES stats file, return cfSite_number, forcing_model, month, and experiment from filename.
"""
function parse_les_path(les_path)
    fname = basename(les_path)
    # The filename usually looks like "Output.cfsite10_HadGEM2-A_amip_2004-2008.01.4x"
    # or "Fields.cfsite10_HadGEM2-A_amip_2004-2008.01.432000.nc"
    fname_split = split(fname, ('.', '_'))
    
    # The site number part is usually the second element when splitting by '.' and '_'
    # (after "Output" or "Fields") e.g. "cfsite10"
    site_part = fname_split[2]
    # Robustly extract only digits from the site part
    cfSite_number = parse(Int64, filter(isdigit, site_part))
    
    forcing_model = fname_split[3]
    experiment = fname_split[4]
    # Month is usually the 6th element in the split
    month = parse(Int64, fname_split[6])
    
    return (cfSite_number, forcing_model, month, experiment)
end

function valid_lespath(les_path)
    # Ensure correct parsing and type handling for dictionary lookups
    cfSite_number, forcing_model, month_int, experiment = parse_les_path(les_path)
    month_str = string(month_int, pad = 2)
    cfSite_number_str = string(cfSite_number, pad = 2)
    
    LES_library = get_LES_library()
    @assert forcing_model in keys(LES_library) "Forcing model $(forcing_model) not valid."
    @assert month_str in keys(LES_library[forcing_model]) "Month $(month_str) not available for $(forcing_model)."
    
    # LES_library[model][month]["cfSite_numbers"] is a Dict{String, String}
    @assert cfSite_number_str in keys(LES_library[forcing_model][month_str]["cfSite_numbers"]) "cfSite $(cfSite_number_str) not found for $(forcing_model), month $(month_str)."
    @assert experiment in LES_library[forcing_model][month_str]["experiments"] "Experiment $(experiment) not available."
end

function get_cfSite_type(cfSite_number::Int)
    if cfSite_number in CFSITE_TYPES["shallow"]
        return "shallow"
    elseif cfSite_number in CFSITE_TYPES["deep"]
        return "deep"
    else
        @error "cfSite number $(cfSite_number) not found in available sites."
    end
end
function get_cfSite_type(i, cfSite_numbers)
    return get_cfSite_type(cfSite_numbers[i])
end

"""
    get_cfSite_les_dir(cfSite_number; forcing_model, month, experiment)

Returns the direct path to the LES directory on the HPC, avoiding any rsync logic.
"""
function get_cfSite_les_dir(
    cfSite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
)
    month_str = string(month, pad = 2)
    cf_str    = string(cfSite_number)
    root_dir  = "/resnick/groups/esm/zhaoyi/GCMForcedLES/cfsite/$(month_str)/$(forcing_model)/$(experiment)/"
    rel_dir   = join(
        [
            "Output.cfsite$(cf_str)",
            forcing_model,
            experiment,
            "2004-2008.$(month_str).4x",
        ],
        "_",
    )
    les_dir = joinpath(root_dir, rel_dir)
    # Check if lespath is valid in our library dictionary
    valid_lespath(les_dir)
    return les_dir
end

"""
    get_stats_path(dir)

Given directory to standard LES or SCM output, fetch path to stats file.
"""
function get_stats_path(dir)
    stats_dir = joinpath(dir, "stats")
    search_dir = ispath(stats_dir) ? stats_dir : dir
    
    try
        stat_files = filter(f -> endswith(f, ".nc"), readdir(search_dir, join = true))
        if length(stat_files) == 1
            return stat_files[1]
        elseif length(stat_files) > 1
            @error "Multiple stats files found in $(search_dir)."
        else
            @warn "No stats file found in $(search_dir). Extending search."
            stat_files = readdir(search_dir, join = true)
            if length(stat_files) == 1
                return stat_files[1]
            else
                @error "No unique stats file found at $(search_dir). Returns $(length(stat_files)) results."
            end
        end
    catch e
        @warn "An error occurred retrieving the stats path at $(search_dir)."
        throw(e)
    end
end

"""
    get_les_calibration_library(; max_cases = 120, models = "HadGEM2-A")

Collect AMIP LES stats paths and cfSite numbers across shallow and deep cases.
"""
function get_les_calibration_library(; max_cases = 120, models = "HadGEM2-A")
    les_library = get_LES_library()

    models_iter = models === nothing ? collect(keys(les_library)) :
                 (isa(models, AbstractString) ? [models] : collect(models))
    for m in models_iter
        @assert haskey(les_library, m) "Model $(m) not found in LES library."
    end

    ref_dirs = []
    cfSite_numbers = Int[]
    i = 0
    for model in models_iter
        for month in keys(les_library[model])
            cfSite_numbers_month = map(
                k -> parse(Int, k),
                collect(keys(les_library[model][month]["cfSite_numbers"])),
            )
            
            les_kwargs = (
                forcing_model = model,
                month = parse(Int, month),
                experiment = "amip",
            )

            cfSite_numbers_month = cfSite_numbers_month[1:min(max_cases-i, lastindex(cfSite_numbers_month))]
            i += length(cfSite_numbers_month)

            paths_for_month = [
                get_stats_path(
                    get_cfSite_les_dir(cfSite_number; les_kwargs...),
                ) for cfSite_number in cfSite_numbers_month
            ]
            append!(ref_dirs, paths_for_month)
            append!(cfSite_numbers, cfSite_numbers_month)

            if i == max_cases
                continue
            end
        end
    end

    if max_cases !== nothing
        n = min(max_cases, length(ref_dirs))
        ref_dirs = ref_dirs[1:n]
        cfSite_numbers = cfSite_numbers[1:n]
    end
    return (ref_dirs, cfSite_numbers)
end

"""
    get_fields_dir(les_dir; timestamp="432000")

Retrieves the directory of raw 3D fields chunks for a particular timestamp.
"""
function get_fields_dir(les_dir::String; timestamp::String="432000")
    fields_dir = joinpath(les_dir, "fields", timestamp)
    if !ispath(fields_dir)
        @warn "Fields directory not found at $(fields_dir)"
    end
    return fields_dir
end

"""
    get_combined_field_path(cfSite_number; forcing_model, month, experiment, timestamp="432000")

Retrieves the path to the post-processed single-file 3D NetCDF.
"""
function get_combined_field_path(
    cfSite_number::Integer;
    forcing_model::String = "HadGEM2-A",
    month::Integer = 7,
    experiment::String = "amip",
    timestamp::String = "432000"
)
    month_str = string(month, pad = 2)
    cf_str    = string(cfSite_number)
    
    # Combined fields are generally placed adjacent to the specific run's Output folder.
    combined_dir = "/resnick/groups/esm/zhaoyi/GCMForcedLES/cfsite/$(month_str)/$(forcing_model)/$(experiment)/fields_combined"
    
    file_name = join(
        [
            "Fields.cfsite$(cf_str)",
            forcing_model,
            experiment,
            "2004-2008.$(month_str).$(timestamp).nc"
        ],
        "_",
    )
    
    path = joinpath(combined_dir, file_name)
    if !isfile(path)
        @error "Combined field not found: $(path)"
    end
    return path
end

"""
    load_4d_fields(les_dir::String, vars::Vector{String}; verbose=true)

Reads all 3D `.nc` files in `les_dir/fields/` across all available timestamps, 
reconstructs them into 4D arrays `(nx, ny, nz, nt)`, and extracts base pressure `p0`.
Returns a `NamedTuple` of `DimArray`s (X, Y, Z, Ti dimensions) plus the scalar/1D profile arrays.
"""
function load_4d_fields(les_dir::String, vars::Vector{String}; verbose=true)
    fields_base = joinpath(les_dir, "fields")
    if !ispath(fields_base)
        error("Fields directory not found in $(fields_base)")
    end
    
    # Auto-detect all timestamp subdirectories
    timestamps_str = filter(x -> tryparse(Int, x) !== nothing, readdir(fields_base))
    timestamps = sort([parse(Int, ts) for ts in timestamps_str])
    
    if isempty(timestamps)
        error("No numeric timestamp directories found in $(fields_base)")
    end
    
    if verbose
        println("Found $(length(timestamps)) timestamps: $(timestamps)")
    end
    
    # Extract base coordinates from the first valid .nc file
    first_time_dir = joinpath(fields_base, string(timestamps[1]))
    first_nc_files = filter(f -> endswith(f, ".nc"), readdir(first_time_dir, join=true))
    
    if isempty(first_nc_files)
        error("No .nc files found in $(first_time_dir)")
    end
    
    nx, ny, nz = 0, 0, 0
    x_coords, y_coords, z_coords = Float64[], Float64[], Float64[]
    
    NC.NCDataset(first_nc_files[1], "r") do ds
        dims_grp = ds.group["dims"]
        nx = dims_grp.dim["x"]
        ny = dims_grp.dim["y"]
        nz = dims_grp.dim["z"]
        x_coords = collect(dims_grp["x"][:])
        y_coords = collect(dims_grp["y"][:])
        z_coords = collect(dims_grp["z"][:])
    end
    
    nt = length(timestamps)
    
    # Pre-allocate 4D arrays
    # Create dictionary mapping string name to the full 4D Float64 Array
    data_arrays = Dict{String, Array{Float64, 4}}()
    for v in vars
        data_arrays[v] = zeros(Float64, nx, ny, nz, nt)
    end
    
    # Loop over time and MPI ranks to fill the chunks
    for (t_idx, ts) in enumerate(timestamps)
        time_dir = joinpath(fields_base, string(ts))
        nc_files = filter(f -> endswith(f, ".nc"), readdir(time_dir, join=true))
        
        for file in nc_files
            NC.NCDataset(file, "r") do ds
                dims_grp = ds.group["dims"]
                fields_grp = ds.group["fields"]
                
                n_i = dims_grp["nl_0"][1]
                n_j = dims_grp["nl_1"][1]
                n_k = dims_grp["nl_2"][1]
                
                # PyCLES limits format 0-indexed values
                i_lo = dims_grp["indx_lo_0"][1] + 1
                j_lo = dims_grp["indx_lo_1"][1] + 1
                k_lo = dims_grp["indx_lo_2"][1] + 1
                
                for v in vars
                    if haskey(fields_grp, v)
                        local_1d = fields_grp[v][:]
                        if length(local_1d) != n_i * n_j * n_k
                            error("Field $v size $(length(local_1d)) does not match expected $(n_i)x$(n_j)x$(n_k)")
                        end
                        local_3d = permutedims(reshape(local_1d, (n_k, n_j, n_i)), (3, 2, 1))
                        data_arrays[v][
                            i_lo : i_lo + n_i - 1, 
                            j_lo : j_lo + n_j - 1, 
                            k_lo : k_lo + n_k - 1, 
                            t_idx
                        ] .= local_3d
                    elseif verbose
                        @warn "Variable $(v) not found in $(file)."
                    end
                end
            end
        end
    end
    
    # Extract pressure profiles from Stats
    stats_file = get_stats_path(les_dir)
    p0_array = Float64[]
    
    try
        NC.NCDataset(stats_file, "r") do ds
            # PyCLES often stores p0 in `profiles`
            if haskey(ds.group, "profiles") && haskey(ds.group["profiles"], "p0")
                p0_array = collect(ds.group["profiles"]["p0"][:])
            elseif haskey(ds.group, "reference") && haskey(ds.group["reference"], "p0")
                p0_array = collect(ds.group["reference"]["p0"][:])
            elseif haskey(ds, "p0")
                p0_array = collect(ds["p0"][:])
            end
        end
    catch e
        @warn "Could not read p0 profile from stats file: $(e)"
    end
    
    # Apply DimensionalData labels
    XDim = DD.X(x_coords)
    YDim = DD.Y(y_coords)
    ZDim = DD.Z(z_coords)
    TDim = DD.Ti(timestamps)

    output_list = Pair{Symbol, Any}[]
    
    for v in vars
        push!(output_list, Symbol(v) => DD.DimArray(data_arrays[v], (XDim, YDim, ZDim, TDim), name=Symbol(v)))
    end
    
    if !isempty(p0_array)
        push!(output_list, :p0 => DD.DimArray(p0_array, (DD.Z(z_coords),), name=:p0))
    end
    
    return DD.DimStack(NamedTuple(output_list))
end

using ..DatasetBuilder: DatasetBuilder
import Arrow

"""
    build_tabular(args...; kwargs...)

Function stub for the tabular building orchestrator. 
Implementation is provided in `utils/build_training_data.jl`.
"""
function build_tabular end

end # module cfSites