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

    The vertical is more complicated, since vertical grids may not be uniform (luckily for us it seems they (mostly) are) Let our finest resolution be 10m, and proceed to up to a 16x reduction from the LES.
    Actually i think we should go until the coarsest reduction is 400m resolution. So if we have a 6000m domain, and say dz = 10m fixed, we can do 10m, 20m, 40m, 80m, 160m, 320m. The next would be 640m, which is larger than 400m, so we stop there. Note that if the native grid is already coarser than 10m, we just start from the native grid and do binary reductions from there until we hit 400m.
    On variable/strtch grids this gets tricker so we should write the code to be robust...

    So in the example above we'd have 3 horizontal resolutions, and 6 vertical resolutions, so 18 different combinations in total.

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


using .GoogleLES: GoogleLES
using .cfSites: cfSites
using .DatasetBuilder: DatasetBuilder
using .CoarseGraining: CoarseGraining
using HTTP: HTTP
using DataFrames: DataFrames
using Arrow: Arrow
using Dates: Dates

const GOOGLELES_FIELD_SPECS = (
    ("q_c", "q_c"),
    ("T", "ta"),
    ("q_t", "hus"),
    ("u", "ua"),
    ("v", "va"),
    ("w", "wa"),
    ("p_ref", "pfull"),
    ("rho", "rhoa"),
    ("theta_li", "thetali"),
)

const CF_SITES_TRANSLATION = (
    ("temperature", "ta"),
    ("qt", "hus"),
    ("u", "ua"),
    ("v", "va"),
    ("w", "wa"),
    ("p", "pfull"),
    ("rho", "rhoa"),
    ("ql", "clw"),
    ("qi", "cli"),
    ("thetali", "thetali"),
)

const GOOGLELES_FULLCASE_BYTES_LIMIT = Int(1_000_000_000)
const GOOGLELES_DEFAULT_BATCH_SIZE = 8

function _parse_bool_env(name::String, default::Bool)
    raw = get(ENV, name, default ? "1" : "0")
    return lowercase(strip(raw)) in ("1", "true", "yes", "y", "on")
end

function case_arrow_filename(site_id::Int, month::Int, experiment::String)
    site_id_str = lpad(string(site_id), 3, '0')
    month_str = lpad(string(month), 2, '0')
    return "googleles_case__$(site_id_str)__month__$(month_str)__exp__$(experiment).arrow"
end

function case_arrow_path(site_id::Int, month::Int, experiment::String, output_dir::String)
    return joinpath(output_dir, case_arrow_filename(site_id, month, experiment))
end

function _empty_googleles_case_df()
    return DataFrames.DataFrame(
        qt = Float32[],
        theta_li = Float32[],
        ta = Float32[],
        p = Float32[],
        rho = Float32[],
        w = Float32[],
        q_liq = Float32[],
        q_ice = Float32[],
        q_con = Float32[],
        liq_fraction = Float32[],
        ice_fraction = Float32[],
        cloud_fraction = Float32[],
        tke = Float32[],
        var_qt = Float32[],
        var_ql = Float32[],
        var_qi = Float32[],
        var_w = Float32[],
        var_h = Float32[],
        cov_qt_ql = Float32[],
        cov_qt_qi = Float32[],
        cov_qt_w = Float32[],
        cov_qt_h = Float32[],
        cov_ql_qi = Float32[],
        cov_ql_w = Float32[],
        cov_ql_h = Float32[],
        cov_qi_w = Float32[],
        cov_qi_h = Float32[],
        cov_w_h = Float32[],
        resolution_h = Float32[],
        domain_h = Float32[],
        resolution_z = Float32[],
        data_source = String[],
        month = Int[],
        cfSite_number = Int[],
        forcing_model = String[],
        experiment = String[],
    )
end

function _estimate_googleles_case_bytes(ds)
    estimated_bytes = 0
    for (g_var, _) in GOOGLELES_FIELD_SPECS
        estimated_bytes += prod(size(ds[g_var])) * sizeof(Float32)
    end
    return estimated_bytes
end

function _progress_print(prefix::AbstractString, current::Int, total::Int, label::AbstractString, started_at::Float64)
    if !(stdout isa Base.TTY || isinteractive())
        return
    end

    width = 26
    total = max(total, 1)
    ratio = clamp(current / total, 0.0, 1.0)
    filled = min(width, max(0, round(Int, ratio * width)))
    empty = width - filled
    bar = repeat("=", max(filled - (current < total ? 1 : 0), 0)) * (current < total ? ">" : "") * repeat(" ", max(empty, 0))
    elapsed = time() - started_at
    rate = current > 0 ? elapsed / current : 0.0
    eta = current < total && rate > 0 ? (total - current) * rate : 0.0
    pct = lpad(string(round(Int, ratio * 100)), 3)
    eta_text = eta > 0 ? string(round(eta; digits=1), "s") : "0.0s"
    print(stdout, '\r', "[", bar, "] ", current, "/", total, " ", pct, "% ", prefix, " ", label, " eta=", eta_text, "   ")
    flush(stdout)
end

function _progress_finish()
    if stdout isa Base.TTY || isinteractive()
        println(stdout)
    end
end

"""
    _slice_t_range_4d(var, timestep_range, t_idx)

Slice a 4D Zarr variable along whichever Julia axis corresponds to logical time `t`.
Returns a view with the time axis retained.
"""
@inline function _slice_t_range_4d(var, timestep_range, t_idx::Int)
    if t_idx < 1 || t_idx > 4
        error("Invalid time axis index: $t_idx")
    end
    return selectdim(var, t_idx, timestep_range)
end

"""
    _perm_to_txyz(julia_dim_names)

Build the permutation that maps Julia axis order to this pipeline's internal
standard order `(t, x, y, z)`.
"""
@inline function _find_dim_idx(julia_dim_names::NTuple{4, Symbol}, target::Symbol)
    @inbounds for i in 1:4
        if julia_dim_names[i] == target
            return i
        end
    end
    return 0
end

function _perm_to_txyz(julia_dim_names::NTuple{4, Symbol})
    t_idx = _find_dim_idx(julia_dim_names, :t)
    x_idx = _find_dim_idx(julia_dim_names, :x)
    y_idx = _find_dim_idx(julia_dim_names, :y)
    z_idx = _find_dim_idx(julia_dim_names, :z)

    if t_idx == 0 || x_idx == 0 || y_idx == 0 || z_idx == 0
        error("Missing one of required dims (t,x,y,z); found dims=$(julia_dim_names)")
    end

    return (t_idx, x_idx, y_idx, z_idx)
end

"""
    _reorder_to_txyz_view(raw, julia_dim_names)

Return a lazy axis-permuted view of `raw` in this pipeline's internal `(t, x, y, z)` order.
"""
@inline function _reorder_to_txyz_view(raw, julia_dim_names::NTuple{4, Symbol})
    perm = _perm_to_txyz(julia_dim_names)
    return PermutedDimsArray(raw, perm)
end

function _load_googleles_cache(ds, timestep_range; field_specs=GOOGLELES_FIELD_SPECS)
    cache = Dict{String, AbstractArray{Float32, 4}}()
    for (g_var, c_var) in field_specs
        var = ds[g_var]
        if !haskey(var.attrs, "_ARRAY_DIMENSIONS")
            error("GoogleLES variable '$g_var' is missing _ARRAY_DIMENSIONS metadata")
        end

        metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
        if length(metadata_dim_names_raw) != 4
            error("GoogleLES variable '$g_var' expected 4 dims, found $(length(metadata_dim_names_raw))")
        end

        julia_dim_names = (
            Symbol(metadata_dim_names_raw[4]),
            Symbol(metadata_dim_names_raw[3]),
            Symbol(metadata_dim_names_raw[2]),
            Symbol(metadata_dim_names_raw[1]),
        )

        t_idx = _find_dim_idx(julia_dim_names, :t)
        if t_idx == 0
            error("GoogleLES variable '$g_var' missing required dim 't'; found dims=$(julia_dim_names)")
        end
        raw = _slice_t_range_4d(var, timestep_range, t_idx)
        canonical = _reorder_to_txyz_view(raw, julia_dim_names)

        if !(eltype(canonical) <: Float32)
            error("GoogleLES variable '$g_var' must be Float32, found eltype=$(eltype(canonical)). Refusing implicit conversion to avoid large allocations.")
        end
        cache[c_var] = canonical
    end
    return cache
end

"""
    _load_googleles_timestep_fields!(cache, ds, timestep_idx; field_specs=GOOGLELES_FIELD_SPECS, z_range=nothing)

Load one GoogleLES timestep for selected fields into `cache` in canonical `(x, y, z)` order.

- For `Float32` source data, values are stored as views (no dense field allocation).
- For non-`Float32` source data, conversion to `Float32` allocates.
- When `z_range` is provided, only that contiguous span is exposed.
"""
function _load_googleles_timestep_fields!(cache::Dict{String, AbstractArray{Float32, 3}}, ds, timestep_idx::Int; field_specs=GOOGLELES_FIELD_SPECS, z_range::Union{Nothing, UnitRange{Int}}=nothing)
    empty!(cache)
    for (g_var, c_var) in field_specs
        var = ds[g_var]
        if !haskey(var.attrs, "_ARRAY_DIMENSIONS")
            error("GoogleLES variable '$g_var' is missing _ARRAY_DIMENSIONS metadata")
        end

        metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
        if length(metadata_dim_names_raw) != 4
            error("GoogleLES variable '$g_var' expected 4 dims, found $(length(metadata_dim_names_raw))")
        end

        julia_dim_names = (
            Symbol(metadata_dim_names_raw[4]),
            Symbol(metadata_dim_names_raw[3]),
            Symbol(metadata_dim_names_raw[2]),
            Symbol(metadata_dim_names_raw[1]),
        )

        t_axis_idx = _find_dim_idx(julia_dim_names, :t)
        if t_axis_idx == 0
            error("GoogleLES variable '$g_var' missing required dim 't'; found dims=$(julia_dim_names)")
        end

        raw = _slice_t_range_4d(var, timestep_idx:timestep_idx, t_axis_idx)
        canonical = _reorder_to_txyz_view(raw, julia_dim_names)
        full_t_slice = @view canonical[1, :, :, :]

        z_view = if isnothing(z_range)
            full_t_slice
        else
            (first(z_range) >= 1 && last(z_range) <= size(full_t_slice, 3)) || error("z_range $(z_range) out of bounds for '$g_var'")
            @view full_t_slice[:, :, z_range]
        end

        if !(eltype(z_view) <: Float32)
            error("GoogleLES variable '$g_var' must be Float32, found eltype=$(eltype(z_view)). Refusing implicit conversion to avoid large allocations.")
        end
        cache[c_var] = z_view
    end
    return cache
end

function _load_googleles_timestep_fields(ds, timestep_idx::Int; field_specs=GOOGLELES_FIELD_SPECS, z_range::Union{Nothing, UnitRange{Int}}=nothing)
    cache = Dict{String, AbstractArray{Float32, 3}}()
    return _load_googleles_timestep_fields!(cache, ds, timestep_idx; field_specs=field_specs, z_range=z_range)
end

"""
    _load_googleles_timestep_fields_into!(dest, ds, timestep_idx; field_specs=GOOGLELES_FIELD_SPECS, z_range, scratch)

Load selected fields for one timestep and copy the requested contiguous `z_range` into
preallocated local buffers `dest[c_var]`.

This is the no-allocation hot-path for span processing:
- minimal remote read (only requested z-range)
- calculations happen on local materialized arrays
- no per-call output array allocation
"""
function _load_googleles_timestep_fields_into!(
    dest::Dict{String, Array{Float32, 3}},
    ds,
    timestep_idx::Int;
    field_specs=GOOGLELES_FIELD_SPECS,
    z_range::UnitRange{Int},
    scratch::Dict{String, AbstractArray{Float32, 3}},
)
    for (_, c_var) in field_specs
        haskey(dest, c_var) || error("Missing destination buffer for '$c_var'")
    end
    empty!(scratch)
    _load_googleles_timestep_fields!(scratch, ds, timestep_idx; field_specs=field_specs, z_range=z_range)
    for (_, c_var) in field_specs
        copyto!(@view(dest[c_var][:, :, z_range]), scratch[c_var])
    end
    return dest
end

"""
    _foreach_true_span(mask, f)

Call `f(k_start, k_end)` for each contiguous `true` run in `mask`.
"""
function _foreach_true_span(mask::BitVector, f::Function)
    n = length(mask)
    k = 1
    while k <= n
        while k <= n && !mask[k]
            k += 1
        end
        k > n && break
        k_start = k
        while k <= n && mask[k]
            k += 1
        end
        f(k_start, k - 1)
    end
    return nothing
end

@inline _foreach_true_span(f::Function, mask::BitVector) = _foreach_true_span(mask, f)

function _has_cloud_after_2x2(q_c::AbstractArray{<:Real, 3}; threshold::Float32=1f-10)
    nx, ny, nz = size(q_c)
    nxc = div(nx, 2)
    nyc = div(ny, 2)
    @inbounds for k in 1:nz
        for j in 1:nyc
            fj = 2j - 1
            for i in 1:nxc
                fi = 2i - 1
                mean_qc = 0.25f0 * (
                    q_c[fi, fj, k] + q_c[fi + 1, fj, k] +
                    q_c[fi, fj + 1, k] + q_c[fi + 1, fj + 1, k]
                )
                if mean_qc >= threshold
                    return true
                end
            end
        end
    end
    return false
end

function _safe_close_http_pools!()
    # Closing HTTP pools can race with background idle-monitor tasks in some
    # HTTP/OpenSSL versions and produce noisy "stream not initialized" errors.
    # Keep this opt-in for now.
    close_pools = lowercase(strip(get(ENV, "MLCD_CLOSE_HTTP_POOLS", "0"))) in ("1", "true", "yes", "y", "on")
    if !close_pools
        return
    end

    try
        HTTP.Connections.closeall()
    catch err
        @debug "HTTP pool cleanup failed" exception=(err, catch_backtrace())
    end
end

function _assert_finite_dataframe(df::DataFrames.DataFrame, context::AbstractString)
    bad_counts = Pair{Symbol, Int}[]
    for nm in names(df)
        col = df[!, nm]
        if !(eltype(col) <: Real)
            continue
        end
        bad = count(x -> !isfinite(x), col)
        if bad > 0
            push!(bad_counts, Symbol(nm) => bad)
        end
    end

    if isempty(bad_counts)
        return
    end

    sort!(bad_counts; by=last, rev=true)
    top = bad_counts[1:min(end, 8)]
    error("Non-finite values detected in generated table ($context). Top bad columns: $(top)")
end

const GOOGLELES_BATCH_SPECS = (
    ("q_c", "q_c"),
    ("T", "ta"),
    ("q_t", "hus"),
    ("u", "ua"),
    ("v", "va"),
    ("w", "wa"),
    ("p_ref", "pfull"),
    ("rho", "rhoa"),
    ("theta_li", "thetali"),
)

function _progress_line(message::AbstractString)
    if isinteractive() || stdout isa Base.TTY
        print(stdout, "\r", message, " ")
        flush(stdout)
    else
        @info message
    end
end

function _progress_done()
    if isinteractive() || stdout isa Base.TTY
        println(stdout)
    end
end

# ------------------------------------------------------------------- #
# Google LES Implementation
# ------------------------------------------------------------------- #

function GoogleLES._build_tabular_legacy(site_id::Int, month::Int, experiment::String, output_dir::String; max_timesteps::Int=0, timestep_batch_size::Int=0, min_h_resolution::Float32=1000.0f0, verbose::Bool=false)
    mkpath(output_dir)
    println("Processing GoogleLES case site=$(site_id) month=$(month) experiment=$(experiment)")
    
    ds = GoogleLES.load_zarr_simulation(site_id, month, experiment)
    if isnothing(ds)
        @error "Could not load simulation."
        _safe_close_http_pools!()
        return
    end

    # Extract dimensions
    nt = length(ds["t"])
    if max_timesteps > 0
        nt = min(nt, max_timesteps)
    end

    estimated_case_bytes = _estimate_googleles_case_bytes(ds)
    fullcase_bytes_limit = parse(Int, get(ENV, "MLCD_GOOGLELES_FULLCASE_BYTES_LIMIT", string(GOOGLELES_FULLCASE_BYTES_LIMIT)))
    force_fullcase = _parse_bool_env("MLCD_GOOGLELES_FORCE_FULLCASE", true)
    effective_batch_size = if timestep_batch_size > 0
        timestep_batch_size
    elseif force_fullcase
        nt
    elseif estimated_case_bytes <= fullcase_bytes_limit
        nt
    else
        min(GOOGLELES_DEFAULT_BATCH_SIZE, nt)
    end
    effective_batch_size = max(1, effective_batch_size)
    is_full_case = effective_batch_size >= nt
    
    x_coords = collect(ds["x"][:])
    z_coords = collect(ds["z"][:])
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    domain_h = x_coords[end] - x_coords[1]
    
    # dz varies with z in some grids, compute profile
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end]) 
    
    metadata = Dict{Symbol, Any}(
        :data_source => "GoogleLES",
        :month => month,
        :site_id => site_id,
        :cfSite_number => site_id,
        :forcing_model => "GoogleLES",
        :experiment => experiment
    )
    
    spatial_info = Dict{Symbol, Any}(
        :dx_native => Float32(dx_native),
        :domain_h => Float32(domain_h),
        :min_h_resolution => Float32(min_h_resolution),
        :dz_native_profile => Float32.(dz_native_profile)
    )

    if verbose
        @info "GoogleLES estimated working set=$(round(estimated_case_bytes / 1_000_000; digits=1)) MB, batch_size=$(effective_batch_size), full_case=$(is_full_case)."
    end

    case_tables = DataFrames.DataFrame[]
    processed_timesteps = 0
    started_at = time()
    processing_seconds = 0.0
    cache_seconds = 0.0

    if is_full_case
        if verbose
            @info "Loading full GoogleLES case into memory for $(site_id)/$(month)/$(experiment)..."
        end
        cache_started_at = time()
        full_cache = _load_googleles_cache(ds, 1:nt)
        cache_elapsed = time() - cache_started_at
        cache_seconds += cache_elapsed
        println("GoogleLES cache load complete for site=$(site_id), month=$(month), nt=$(nt): $(round(cache_elapsed; digits=1))s")

        processing_started_at = time()

        q_c0 = @view full_cache["q_c"][1, :, :, :]
        clw_buf = similar(q_c0)
        cli_buf = similar(q_c0)
        fine_fields = Dict{String, AbstractArray{Float32, 3}}()
        metadata_t = copy(metadata)
        metadata_t[:verbose] = verbose

        # preallocate fine_fields
        for (_, c_var) in GOOGLELES_BATCH_SPECS
            fine_fields[c_var] = similar(q_c0)
        end

        for local_t in 1:nt
            step_started_at = time()
            processed_timesteps += 1
            t_idx = local_t - 1
            _progress_print("GoogleLES", processed_timesteps, nt, "site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)", processing_started_at)
            # Keep timestep data as views so the builder can reuse the same backing cache.
            # q_c = @view full_cache["q_c"][local_t, :, :, :]
            q_c = fine_fields["q_c"]
            fine_fields["q_c"] .= full_cache["q_c"][local_t, :, :, :] # force materialization from remote zarr store for downstream speed
            if !_has_cloud_after_2x2(q_c)
                continue
            end

            fine_fields["q_c"] = q_c
            for (_, c_var) in GOOGLELES_FIELD_SPECS
                if c_var == "q_c"
                    continue
                end
                # fine_fields[c_var] .= @view full_cache[c_var][local_t, :, :, :]
                fine_fields[c_var] .= full_cache[c_var][local_t, :, :, :] 
            end

            ta = fine_fields["ta"]

            @inbounds for i in eachindex(q_c)
                clw_buf[i], cli_buf[i] = GoogleLES.partition_condensate(q_c[i], ta[i])
            end

            fine_fields["clw"] = clw_buf
            fine_fields["cli"] = cli_buf

            metadata_t[:timestep] = t_idx

            df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
            if size(df, 1) > 0
                _assert_finite_dataframe(df, "GoogleLES site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)")
                push!(case_tables, df)
            end

            processing_seconds += (time() - step_started_at)
            if processed_timesteps % 8 == 0 || processed_timesteps == nt
                avg_step = processing_seconds / max(processed_timesteps, 1)
                println("\nGoogleLES processing progress: site=$(site_id), month=$(month), processed=$(processed_timesteps)/$(nt), avg_step_seconds=$(round(avg_step; digits=2))")
            end
        end

        println("GoogleLES timestep processing complete for site=$(site_id), month=$(month): total=$(round(processing_seconds; digits=1))s, per_timestep=$(round(processing_seconds / max(nt, 1); digits=2))s")
    else
        if verbose
            @info "Processing GoogleLES case in batches of $(effective_batch_size); estimated working set exceeds threshold."
        end

        for batch_start in 0:effective_batch_size:(nt - 1)
            batch_stop = min(batch_start + effective_batch_size - 1, nt - 1)
            batch_range = batch_start + 1:batch_stop + 1
            cache_started_at = time()
            batch_cache = _load_googleles_cache(ds, batch_range)
            cache_elapsed = time() - cache_started_at
            cache_seconds += cache_elapsed
            println("GoogleLES batch cache load: site=$(site_id), month=$(month), timesteps=$(first(batch_range)-1):$(last(batch_range)-1), seconds=$(round(cache_elapsed; digits=1))")

            processing_started_at = time()

            q_c_batch = batch_cache["q_c"]
            q_c0 = @view q_c_batch[1, :, :, :]
            clw_buf = similar(q_c0)
            cli_buf = similar(q_c0)
            fine_fields = Dict{String, AbstractArray{Float32, 3}}()
            metadata_t = copy(metadata)
            metadata_t[:verbose] = verbose

            for local_t in 1:size(q_c_batch, 1)
                step_started_at = time()
                processed_timesteps += 1
                t_idx = batch_start + local_t - 1
                _progress_print("GoogleLES", processed_timesteps, nt, "site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)", processing_started_at)
                # Keep timestep data as views so the builder can reuse the same backing cache.
                q_c = @view q_c_batch[local_t, :, :, :]
                if !_has_cloud_after_2x2(q_c)
                    continue
                end

                fine_fields["q_c"] = q_c
                for (_, c_var) in GOOGLELES_FIELD_SPECS
                    if c_var == "q_c"
                        continue
                    end
                    fine_fields[c_var] = @view batch_cache[c_var][local_t, :, :, :]
                end

                ta = fine_fields["ta"]

                @inbounds for i in eachindex(q_c)
                    clw_buf[i], cli_buf[i] = GoogleLES.partition_condensate(q_c[i], ta[i])
                end

                fine_fields["clw"] = clw_buf
                fine_fields["cli"] = cli_buf

                metadata_t[:timestep] = t_idx

                df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
                if size(df, 1) > 0
                    _assert_finite_dataframe(df, "GoogleLES site=$(site_id) month=$(month) experiment=$(experiment) timestep=$(t_idx)")
                    push!(case_tables, df)
                end

                processing_seconds += (time() - step_started_at)
                if processed_timesteps % 8 == 0 || processed_timesteps == nt
                    avg_step = processing_seconds / max(processed_timesteps, 1)
                    println("\nGoogleLES processing progress: site=$(site_id), month=$(month), processed=$(processed_timesteps)/$(nt), avg_step_seconds=$(round(avg_step; digits=2))")
                end
            end
        end

        println("GoogleLES timestep processing complete for site=$(site_id), month=$(month): total=$(round(processing_seconds; digits=1))s, per_timestep=$(round(processing_seconds / max(nt, 1); digits=2))s")
    end

    _progress_finish()

    total_elapsed = cache_seconds + processing_seconds
    if total_elapsed > 0
        cache_pct = 100 * cache_seconds / total_elapsed
        processing_pct = 100 * processing_seconds / total_elapsed
        println("GoogleLES time breakdown: cache=$(round(cache_seconds; digits=1))s ($(round(cache_pct; digits=1))%), processing=$(round(processing_seconds; digits=1))s ($(round(processing_pct; digits=1))%), total=$(round(total_elapsed; digits=1))s")
    end

    if isempty(case_tables)
        out_file = case_arrow_path(site_id, month, experiment, output_dir)
        Arrow.write(out_file, _empty_googleles_case_df())
        if verbose
            @info "No trainable GoogleLES rows found for site=$site_id month=$month experiment=$experiment; wrote empty Arrow case file."
        end
        _safe_close_http_pools!()
        return
    end

    # out_file = joinpath(output_dir, "googleles_case__$(site_id)__month__$(month)__exp__$(experiment).arrow")
    out_file = case_arrow_path(site_id, month, experiment, output_dir)
    final_df = vcat(case_tables...)
    _assert_finite_dataframe(final_df, "GoogleLES final site=$(site_id) month=$(month) experiment=$(experiment)")
    # TODO: Add schema version metadata (requires investigating correct Arrow.jl API)
    Arrow.write(out_file, final_df)
    println("Wrote GoogleLES case file $(out_file) with $(sum(DataFrames.nrow, case_tables)) rows.")
    _safe_close_http_pools!()
end

# ------------------------------------------------------------------- #
# cfSites Implementation
# ------------------------------------------------------------------- #

function cfSites._build_tabular_legacy(cfSite_number::Int, month::Int, forcing_model::String, experiment::String, output_dir::String; max_timesteps::Int=0, min_h_resolution::Float32=1000.0f0, verbose::Bool=false)
    mkpath(output_dir)
    println("Processing cfSites case site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment)")
    
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
    domain_h = x_coords[end] - x_coords[1]
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
        :domain_h => Float32(domain_h),
        :min_h_resolution => Float32(min_h_resolution),
        :dz_native_profile => Float32.(dz_native_profile)
    )
    
    case_tables = DataFrames.DataFrame[]
    processed_timesteps = 0
    started_at = time()
    
    for t_idx in 1:nt
        processed_timesteps += 1
        _progress_print("cfSites", processed_timesteps, nt, "site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment) timestep=$(t_idx)", started_at)
        fine_fields = Dict{String, AbstractArray{Float32, 3}}()

        for (site_var, canonical_var) in CF_SITES_TRANSLATION
            fine_fields[canonical_var] = Float32.(ds_stack[Symbol(site_var)][DimensionalData.Ti(t_idx)])
        end

        metadata_t = copy(metadata)
        metadata_t[:timestep] = t_idx

        df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
        if size(df, 1) > 0
            _assert_finite_dataframe(df, "cfSites site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment) timestep=$(t_idx)")
            push!(case_tables, df)
        end
    end

    _progress_finish()

    if isempty(case_tables)
        if verbose
            @info "No trainable cfSites rows found for site=$cfSite_number month=$month model=$forcing_model experiment=$experiment; no Arrow file written."
        end
        return
    end

    out_file = joinpath(output_dir, "cfsites_$(forcing_model)_$(experiment)_$(month)_$(cfSite_number).arrow")
    final_df = vcat(case_tables...)
    _assert_finite_dataframe(final_df, "cfSites final site=$(cfSite_number) month=$(month) model=$(forcing_model) experiment=$(experiment)")
    # TODO: Add schema version metadata (requires investigating correct Arrow.jl API)
    Arrow.write(out_file, final_df)
    println("Wrote cfSites case file $(out_file) with $(sum(DataFrames.nrow, case_tables)) rows.")
end
