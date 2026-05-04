"""
Profile GoogleLES with Julia's built-in profilers.

This script intentionally reports compact flat summaries instead of dumping
raw traces. It profiles two scopes:
- the full end-to-end generation pipeline, and
- the isolated `process_abstract_chunk` kernel.

Preferred REPL usage:
    using Pkg
    Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
    include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")

Terminal fallback:
    julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline -e 'include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile_googleles_pipeline.jl")'

Optional environment variables:
    SITE_ID=343
    MONTH=1
    EXPERIMENT=amip
    PROFILE_SCOPE=both|pipeline|chunk
    PROFILE_TIMESTEPS=8
    PROFILE_CHUNK_REPEATS=25
    PROFILE_SAMPLE_RATE=0.01
    PROFILE_MINCOUNT=10
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)

using MLCondensateDistributions: MLCondensateDistributions as MLCD
using Dates: Dates
using Profile: Profile
using Profile.Allocs

const SITE_ID = parse(Int, get(ENV, "SITE_ID", "343"))
const MONTH = parse(Int, get(ENV, "MONTH", "1"))
const EXPERIMENT = get(ENV, "EXPERIMENT", "amip")
const PROFILE_SCOPE = lowercase(get(ENV, "PROFILE_SCOPE", "both"))
const PROFILE_TIMESTEPS = max(1, parse(Int, get(ENV, "PROFILE_TIMESTEPS", "8")))
const PROFILE_CHUNK_REPEATS = max(1, parse(Int, get(ENV, "PROFILE_CHUNK_REPEATS", "25")))
const PROFILE_SAMPLE_RATE = parse(Float64, get(ENV, "PROFILE_SAMPLE_RATE", "0.01"))
const PROFILE_MINCOUNT = max(1, parse(Int, get(ENV, "PROFILE_MINCOUNT", "10")))
const PROFILE_OUT_DIR = abspath(get(ENV, "PROFILE_OUT_DIR", joinpath(@__DIR__, "profile_reports")))
const PROFILE_RUN_TAG = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
const PROFILE_RUN_DIR = joinpath(PROFILE_OUT_DIR, "profile_$(PROFILE_RUN_TAG)")

function _slug(s::String)
    return replace(lowercase(s), r"[^a-z0-9]+" => "_")
end

function _write_report(path::String, writer::Function)
    mkpath(dirname(path))
    open(path, "w") do io
        writer(io)
    end
    println("Wrote profile report: $(path)")
end

function _build_context(ds, site_id, month, experiment)
    x_coords = collect(ds["x"][:])
    z_coords = collect(ds["z"][:])
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    domain_h = x_coords[end] - x_coords[1]
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end])

    spatial_info = Dict{Symbol, Any}(
        :dx_native => Float32(dx_native),
        :domain_h => Float32(domain_h),
        :min_dh => 1000.0f0,
        :dz_native_profile => Float32.(dz_native_profile),
    )

    metadata = Dict{Symbol, Any}(
        :data_source => "GoogleLES",
        :month => month,
        :site_id => site_id,
        :cfSite_number => site_id,
        :forcing_model => "GoogleLES",
        :experiment => experiment,
        :timestep => 0,
    )

    return metadata, spatial_info
end

function _build_chunk_inputs(cache, local_t::Int, metadata, spatial_info)
    q_c = @view cache["q_c"][local_t, :, :, :]
    cg_qc = MLCD.CoarseGraining.cg_2x2_horizontal(q_c)
    if all(cg_qc .< 1f-8)
        return nothing
    end

    fine_fields = Dict{String, AbstractArray{Float32, 3}}()
    for c_var in ("ta", "hus", "ua", "va", "wa", "pfull", "rhoa", "thetali")
        fine_fields[c_var] = @view cache[c_var][local_t, :, :, :]
    end

    ta = fine_fields["ta"]
    clw = similar(q_c)
    cli = similar(q_c)
    @inbounds for i in eachindex(q_c)
        clw[i], cli[i] = MLCD.GoogleLES.partition_condensate(q_c[i], ta[i])
    end
    fine_fields["clw"] = clw
    fine_fields["cli"] = cli

    metadata_t = copy(metadata)
    metadata_t[:timestep] = local_t - 1
    return fine_fields, metadata_t, spatial_info
end

function _print_time_profile(label::String, f::Function)
    Profile.clear()
    Profile.@profile f()
    println("\n== $(label) : time ==")
    Profile.print(IOContext(stdout, :displaysize => (24, 160)); format=:flat, sortedby=:count, mincount=PROFILE_MINCOUNT)

    out_file = joinpath(PROFILE_RUN_DIR, "$( _slug(label) )_time_flat.txt")
    _write_report(out_file, io -> begin
        println(io, "label=$(label)")
        println(io, "format=flat sortedby=count mincount=$(PROFILE_MINCOUNT)")
        println(io)
        Profile.print(IOContext(io, :displaysize => (200, 220)); format=:flat, sortedby=:count, mincount=PROFILE_MINCOUNT)
    end)
end

function _print_alloc_profile(label::String, f::Function)
    Profile.Allocs.clear()
    f()
    prof = Profile.Allocs.fetch()
    println("\n== $(label) : allocations ==")
    println("samples = ", length(prof.allocs), ", sample_rate = ", PROFILE_SAMPLE_RATE)
    Profile.Allocs.print(IOContext(stdout, :displaysize => (24, 160)), prof; format=:flat, sortedby=:count, mincount=PROFILE_MINCOUNT)

    out_file = joinpath(PROFILE_RUN_DIR, "$( _slug(label) )_allocs_flat.txt")
    _write_report(out_file, io -> begin
        println(io, "label=$(label)")
        println(io, "samples=$(length(prof.allocs)) sample_rate=$(PROFILE_SAMPLE_RATE)")
        println(io, "format=flat sortedby=count mincount=$(PROFILE_MINCOUNT)")
        println(io)
        Profile.Allocs.print(IOContext(io, :displaysize => (200, 220)), prof; format=:flat, sortedby=:count, mincount=PROFILE_MINCOUNT)
    end)
end

function _close_http_pools!()
    if isdefined(MLCD, Symbol("_safe_close_http_pools!"))
        getfield(MLCD, Symbol("_safe_close_http_pools!"))()
    end
end

function _profile_pipeline()
    output_dir = mktempdir()
    println("Profiling end-to-end pipeline into temporary output dir: $(output_dir)")

    profile_one = () -> MLCD.GoogleLES.build_tabular(
        SITE_ID,
        MONTH,
        EXPERIMENT,
        output_dir;
        max_timesteps=PROFILE_TIMESTEPS,
        min_dh=1000.0f0,
        verbose=false,
    )

    # Warm up JIT and HTTP connection setup outside the profile window.
    profile_one()
    _print_time_profile("GoogleLES build_tabular end-to-end", profile_one)
    _print_alloc_profile("GoogleLES build_tabular end-to-end", () -> begin
        Profile.Allocs.@profile sample_rate=PROFILE_SAMPLE_RATE profile_one()
    end)
    _close_http_pools!()
end

function _profile_chunk_kernel()
    println("Profiling isolated process_abstract_chunk kernel")
    ds = MLCD.GoogleLES.load_zarr_simulation(SITE_ID, MONTH, EXPERIMENT)
    if isnothing(ds)
        error("Could not load GoogleLES simulation for profiling")
    end

    load_cache = getfield(MLCD, Symbol("_load_googleles_cache"))
    cache = load_cache(ds, 1:PROFILE_TIMESTEPS)
    metadata, spatial_info = _build_context(ds, SITE_ID, MONTH, EXPERIMENT)
    chunk_inputs = nothing
    for local_t in 1:PROFILE_TIMESTEPS
        chunk_inputs = _build_chunk_inputs(cache, local_t, metadata, spatial_info)
        if !isnothing(chunk_inputs)
            println("Using representative timestep $(local_t) for chunk profiling")
            break
        end
    end
    if isnothing(chunk_inputs)
        @warn "All sampled timesteps were fully masked out; using synthetic cloudy chunk fallback for chunk profiling"
        dims = (16, 16, 16)
        fine_fields = Dict{String, AbstractArray{Float32, 3}}()
        for k in ("ta", "hus", "wa", "ua", "va", "thetali", "pfull", "rhoa")
            fine_fields[k] = rand(Float32, dims...)
        end
        clw = zeros(Float32, dims...)
        cli = zeros(Float32, dims...)
        clw[1:4, 1:4, 5] .= 0.05f0
        cli[1:4, 1:4, 5] .= 0.01f0
        fine_fields["clw"] = clw
        fine_fields["cli"] = cli
        metadata_t = Dict{Symbol, Any}(
            :data_source => "Synth",
            :month => MONTH,
            :cfSite_number => SITE_ID,
            :forcing_model => "GoogleLES",
            :experiment => EXPERIMENT,
            :timestep => 0,
        )
        spatial_info_t = Dict{Symbol, Any}(
            :dx_native => 50.0f0,
            :domain_h => 6000.0f0,
            :min_dh => 1000.0f0,
            :dz_native_profile => fill(25.0f0, dims[3]),
        )
        chunk_inputs = (fine_fields, metadata_t, spatial_info_t)
    end
    fine_fields, metadata_t, spatial_info_t = chunk_inputs

    kernel = () -> begin
        for _ in 1:PROFILE_CHUNK_REPEATS
            MLCD.DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info_t)
        end
    end

    kernel()
    _print_time_profile("process_abstract_chunk repeated kernel", kernel)
    _print_alloc_profile("process_abstract_chunk repeated kernel", () -> begin
        Profile.Allocs.@profile sample_rate=PROFILE_SAMPLE_RATE kernel()
    end)
    _close_http_pools!()
end

println("== GoogleLES Profile ==")
println("site=$(SITE_ID), month=$(MONTH), experiment=$(EXPERIMENT), scope=$(PROFILE_SCOPE)")
println("profile_timesteps=$(PROFILE_TIMESTEPS), chunk_repeats=$(PROFILE_CHUNK_REPEATS), sample_rate=$(PROFILE_SAMPLE_RATE), mincount=$(PROFILE_MINCOUNT)")
println("profile_output_dir=$(PROFILE_RUN_DIR)")
mkpath(PROFILE_RUN_DIR)

if PROFILE_SCOPE in ("both", "pipeline")
    _profile_pipeline()
end

if PROFILE_SCOPE in ("both", "chunk")
    _profile_chunk_kernel()
end

println("\nDone. These are flat profiler summaries, not raw trace dumps.")