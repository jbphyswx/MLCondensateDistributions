"""
Benchmark how remote GoogleLES q_c read time scales with contiguous z-window size.

This script isolates the q_c array and times materialization of a single timestep
for increasing z-window sizes. Because the z chunk size is 1, this directly tests
how much bandwidth is saved when fewer vertical layers are requested.

Usage:
    julia --project=. experiments/amip_baseline/profile/profile_qc_z_window_scaling.jl

Optional env vars:
    SITE_ID=320
    MONTH=1
    EXPERIMENT=amip
    T_IDX=1
    Z_LIST=1,2,4,8,16,32,48,64,73
    REPEATS=3
    WARMUP_PER_Z=true
"""

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions

const SITE_ID = parse(Int, get(ENV, "SITE_ID", "320"))
const MONTH = parse(Int, get(ENV, "MONTH", "1"))
const EXPERIMENT = get(ENV, "EXPERIMENT", "amip")
const T_IDX = parse(Int, get(ENV, "T_IDX", "1"))
const REPEATS = parse(Int, get(ENV, "REPEATS", "3"))
const WARMUP_PER_Z = lowercase(strip(get(ENV, "WARMUP_PER_Z", "true"))) in ("1", "true", "yes", "y", "on")

function parse_int_list(raw::String)
    vals = Int[]
    for token in split(raw, ',')
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    sort!(unique!(vals))
    return vals
end

function median_val(values::Vector{Float64})
    s = sort(values)
    n = length(s)
    if isodd(n)
        return s[(n + 1) >>> 1]
    end
    mid = n >>> 1
    return 0.5 * (s[mid] + s[mid + 1])
end

function main()
    ds = MLCondensateDistributions.GoogleLES.load_zarr_simulation(SITE_ID, MONTH, EXPERIMENT)
    isnothing(ds) && error("Could not load simulation")

    q_c = ds["q_c"]
    nz, nx, ny, nt = size(q_c)
    t_idx = clamp(T_IDX, 1, nt)
    z_default = "1,2,4,8,16,32,64,128,256,320,$(nz)"
    z_list = [z for z in parse_int_list(get(ENV, "Z_LIST", z_default)) if 1 <= z <= nz]
    isempty(z_list) && error("Z_LIST produced no valid z windows in [1,$nz]")

    println("== profile_qc_z_window_scaling ==")
    println("site=", SITE_ID, " month=", MONTH, " experiment=", EXPERIMENT, " timestep=", t_idx)
    println("shape=", (nz, nx, ny, nt), " z_chunk=", getproperty(q_c, :metadata).chunks[1])
    println("z_list=", z_list)
    println("window_layers,load_seconds_med,mb_materialized,ms_per_layer")

    for z_layers in z_list
        load_times = Float64[]
        mat_mb = 0.0
        z_start = 1
        z_stop = z_layers

        if WARMUP_PER_Z
            _ = Array(@view q_c[z_start:z_stop, :, :, t_idx])
        end

        for _ in 1:REPEATS
            load_s = @elapsed begin
                global window = Array(@view q_c[z_start:z_stop, :, :, t_idx])
            end
            push!(load_times, load_s)
            mat_mb = sizeof(window) / 1024.0^2
        end

        med = median_val(load_times)
        ms_per_layer = 1000.0 * med / z_layers
        println(z_layers, ",", round(med; digits=3), ",", round(mat_mb; digits=1), ",", round(ms_per_layer; digits=2))
    end
end

main()
