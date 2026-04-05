"""
Measure processing-only runtime (excluding zarr read/decompression) by
materializing GoogleLES fields into RAM first, then timing old vs implementation builders.

Usage:
    julia --project=. experiments/amip_baseline/profile/profile_processing_only_inmemory.jl

Optional env vars:
    SITE_ID=320
    MONTH=1
    EXPERIMENT=amip
    MAX_TIMESTEPS=2
"""

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD

const SITE_ID = parse(Int, get(ENV, "SITE_ID", "320"))
const MONTH = parse(Int, get(ENV, "MONTH", "1"))
const EXPERIMENT = get(ENV, "EXPERIMENT", "amip")
const MAX_TIMESTEPS = parse(Int, get(ENV, "MAX_TIMESTEPS", "2"))

function build_spatial_info(ds, min_h_resolution)
    x_coords = collect(ds["x"][:])
    z_coords = collect(ds["z"][:])
    dx_native = (x_coords[end] - x_coords[1]) / (length(x_coords) - 1)
    domain_h = x_coords[end] - x_coords[1]
    dz_native_profile = diff(z_coords)
    push!(dz_native_profile, dz_native_profile[end])

    return Dict{Symbol, Any}(
        :dx_native => Float32(dx_native),
        :domain_h => Float32(domain_h),
        :min_h_resolution => Float32(min_h_resolution),
        :dz_native_profile => Float32.(dz_native_profile),
    )
end

function build_metadata(site_id, month, experiment)
    return Dict{Symbol, Any}(
        :data_source => "GoogleLES",
        :month => month,
        :site_id => site_id,
        :cfSite_number => site_id,
        :forcing_model => "GoogleLES",
        :experiment => experiment,
        :verbose => false,
    )
end

function materialize_cache(cache_lazy)
    out = Dict{String, Array{Float32, 4}}()
    for (k, v) in cache_lazy
        out[k] = Array(v)
    end
    return out
end

function run_old_processing_only(cache, nt, metadata, spatial_info)
    q_c0 = @view cache["q_c"][1, :, :, :]
    clw_buf = similar(q_c0)
    cli_buf = similar(q_c0)

    fine_fields = Dict{String, AbstractArray{Float32, 3}}()
    for (_, c_var) in MLCD.GOOGLELES_BATCH_SPECS
        fine_fields[c_var] = similar(q_c0)
    end

    rows = 0
    t0 = time()
    for local_t in 1:nt
        t_idx = local_t - 1

        q_c = fine_fields["q_c"]
        q_c .= cache["q_c"][local_t, :, :, :]
        if !MLCD._has_cloud_after_2x2(q_c)
            continue
        end

        for (_, c_var) in MLCD.GOOGLELES_FIELD_SPECS
            if c_var == "q_c"
                continue
            end
            fine_fields[c_var] .= cache[c_var][local_t, :, :, :]
        end

        ta = fine_fields["ta"]
        @inbounds for i in eachindex(q_c)
            clw_buf[i], cli_buf[i] = MLCD.GoogleLES.partition_condensate(q_c[i], ta[i])
        end

        fine_fields["clw"] = clw_buf
        fine_fields["cli"] = cli_buf

        metadata_t = copy(metadata)
        metadata_t[:timestep] = t_idx
        df = MLCD.DatasetBuilder.process_abstract_chunk(fine_fields, metadata_t, spatial_info)
        rows += nrow(df)
    end
    return time() - t0, rows
end

function run_impl_processing_only(cache, nt, metadata, spatial_info)
    q_c0 = @view cache["q_c"][1, :, :, :]
    q_c_buf = similar(q_c0)
    clw_buf = similar(q_c0) # this allocates
    cli_buf = similar(q_c0) # this allocates

    fine_fields = Dict{String, Array{Float32, 3}}()
    fine_fields["q_c"] = q_c_buf
    for (_, c_var) in MLCD.GOOGLELES_FIELD_SPECS
        if c_var == "q_c"
            continue
        end
        fine_fields[c_var] = similar(q_c0)
    end

    rows = 0
    t0 = time()
    for local_t in 1:nt
        t_idx = local_t - 1

        fine_fields["q_c"] .= cache["q_c"][local_t, :, :, :]
        if !MLCD._has_cloud_after_2x2(fine_fields["q_c"])
            continue
        end

        for (_, c_var) in MLCD.GOOGLELES_FIELD_SPECS
            if c_var == "q_c"
                continue
            end
            fine_fields[c_var] .= cache[c_var][local_t, :, :, :]
        end

        ta = fine_fields["ta"]
        @inbounds for i in eachindex(q_c_buf)
            clw_buf[i], cli_buf[i] = MLCD.GoogleLES.partition_condensate(q_c_buf[i], ta[i])
        end

        fine_fields["clw"] = clw_buf
        fine_fields["cli"] = cli_buf

        metadata_t = copy(metadata)
        metadata_t[:timestep] = t_idx
        df = MLCD.DatasetBuilder.DatasetBuilderImpl.process_abstract_chunk_impl(fine_fields, metadata_t, spatial_info)
        rows += nrow(df)
    end
    return time() - t0, rows
end

function main()
    println("== profile_processing_only_inmemory ==")
    println("site=", SITE_ID, " month=", MONTH, " experiment=", EXPERIMENT, " max_timesteps=", MAX_TIMESTEPS)

    ds = MLCD.GoogleLES.load_zarr_simulation(SITE_ID, MONTH, EXPERIMENT)
    isnothing(ds) && error("Could not load simulation")

    nt = min(length(ds["t"]), MAX_TIMESTEPS)
    metadata = build_metadata(SITE_ID, MONTH, EXPERIMENT)
    spatial_info = build_spatial_info(ds, 1000.0f0)

    t_cache = @elapsed begin
        cache_lazy = MLCD._load_googleles_cache(ds, 1:nt)
        global cache_mem = materialize_cache(cache_lazy)
    end
    println("materialize_to_memory_seconds=", round(t_cache; digits=3), " nt=", nt)

    # warmup
    run_old_processing_only(cache_mem, min(nt, 1), metadata, spatial_info)
    run_impl_processing_only(cache_mem, min(nt, 1), metadata, spatial_info)

    old_s, old_rows = run_old_processing_only(cache_mem, nt, metadata, spatial_info)
    impl_s, impl_rows = run_impl_processing_only(cache_mem, nt, metadata, spatial_info)

    println("old_processing_only_seconds=", round(old_s; digits=3), " rows=", old_rows)
    println("impl_processing_only_seconds=", round(impl_s; digits=3), " rows=", impl_rows)
    println("speed_ratio_old_over_impl_processing_only=", round(old_s / impl_s; digits=3))
end

main()
