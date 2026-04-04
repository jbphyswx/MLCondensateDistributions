"""
Measure how GoogleLES zarr load/materialization time scales with timesteps.

This script times two stages for each timestep count n:
1) metadata+view load via _load_googleles_cache(ds, 1:n)
2) full materialization to RAM via Array(v) for each cached field

Usage:
    julia --project=. experiments/amip_baseline/profile/profile_zarr_load_scaling.jl

Optional env vars:
    SITE_ID=320
    MONTH=1
    EXPERIMENT=amip
    N_LIST=1,2,4,8,16,32,64,73
    REPEATS=2
"""

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD
const SITE_ID = parse(Int, get(ENV, "SITE_ID", "320"))
const MONTH = parse(Int, get(ENV, "MONTH", "1"))
const EXPERIMENT = get(ENV, "EXPERIMENT", "amip")
const REPEATS = parse(Int, get(ENV, "REPEATS", "2"))

function parse_n_list(raw::String)
    vals = Int[]
    for token in split(raw, ',')
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    sort!(unique!(vals))
    return vals
end

function materialize_cache(cache_lazy)
    out = Dict{String, Array{Float32, 4}}()
    for (k, v) in cache_lazy
        out[k] = Array(v)
    end
    return out
end

function median_val(v::Vector{Float64})
    s = sort(v)
    m = length(s)
    if isodd(m)
        return s[(m + 1) >>> 1]
    end
    i = m >>> 1
    return 0.5 * (s[i] + s[i + 1])
end

function main()
    ds = MLCD.GoogleLES.load_zarr_simulation(SITE_ID, MONTH, EXPERIMENT)
    isnothing(ds) && error("Could not load simulation")
    nt_total = length(ds["t"])

    default_n = "1,2,4,8,16,32,64,$(nt_total)"
    n_raw = get(ENV, "N_LIST", default_n)
    n_list = [n for n in parse_n_list(n_raw) if 1 <= n <= nt_total]
    isempty(n_list) && error("N_LIST produced no valid timestep counts in [1,$nt_total]")

    println("== profile_zarr_load_scaling ==")
    println("site=", SITE_ID, " month=", MONTH, " experiment=", EXPERIMENT, " nt_total=", nt_total, " repeats=", REPEATS)
    println("n_list=", n_list)
    println("n,load_s_med,materialize_s_med,total_s_med,gb_materialized,ms_per_timestep_total")

    for n in n_list
        load_times = Float64[]
        mat_times = Float64[]
        mat_gb = 0.0

        for _ in 1:REPEATS
            load_s = @elapsed begin
                global cache_lazy = MLCD._load_googleles_cache(ds, 1:n)
            end
            push!(load_times, load_s)

            mat_s = @elapsed begin
                global cache_mem = materialize_cache(cache_lazy)
            end
            push!(mat_times, mat_s)

            bytes = 0
            for arr in values(cache_mem)
                bytes += sizeof(arr)
            end
            mat_gb = bytes / 1024.0^3
        end

        load_med = median_val(load_times)
        mat_med = median_val(mat_times)
        total_med = load_med + mat_med
        ms_per_t = (1000.0 * total_med) / n

        println(
            n, ",",
            round(load_med; digits=3), ",",
            round(mat_med; digits=3), ",",
            round(total_med; digits=3), ",",
            round(mat_gb; digits=3), ",",
            round(ms_per_t; digits=2),
        )
    end
end

main()
