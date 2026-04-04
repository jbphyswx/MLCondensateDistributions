#!/usr/bin/env julia
# Manual experiment only (not `Pkg.test`): times remote GCS reads for one GoogleLES field using the
# same `_load_googleles_timestep_fields!` path as `build_tabular`.
#
# `_load_googleles_timestep_fields!` only installs **lazy views**; bytes move on materialization
# (`copyto!`). This script times **load + copyto!** into a dense `Array{Float32}` (like the pipeline).
#
# Requires network. Compares full-z vs contiguous z-range vs fused vector z-indices for one timestep.
#
# Usage (from repo root):
#   julia --project=. experiments/amip_baseline/profile/benchmark_remote_zarr_slab_reads.jl [site] [month] [t_1based] [experiment]

const ROOT = dirname(dirname(dirname(@__DIR__)))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD
using Statistics: Statistics
const FIELD_SPECS = (("T", "ta"),)
const N_SAMPLES = 7

function _median_secs(f; n_warmup::Int=2, n::Int=N_SAMPLES)
    for _ in 1:n_warmup
        f()
    end
    samples = Float64[@elapsed f() for _ in 1:n]
    return Statistics.median(samples), samples
end

"""Load views, then force the same remote I/O the tabular path relies on (`copyto!` into dense storage)."""
function _materialize_ta!(cache, ds, t1, z_subset, dest::Array{Float32,3})
    MLCD._load_googleles_timestep_fields!(cache, ds, t1; field_specs=FIELD_SPECS, z_subset=z_subset)
    a = cache["ta"]
    @assert size(dest) == size(a)
    copyto!(dest, a)
    return nothing
end

function main()
    site = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 313
    month = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    t1 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    experiment = length(ARGS) >= 4 ? ARGS[4] : "amip"

    ds = MLCD.GoogleLES.load_zarr_simulation(site, month, experiment)
    ds === nothing && error("Failed to open Zarr (network or path).")

    cache = Dict{String, AbstractArray{Float32, 3}}()

    # Compile + establish nx, ny, nz (materialize once so compile + cache effects land here)
    MLCD._load_googleles_timestep_fields!(cache, ds, t1; field_specs=FIELD_SPECS, z_subset=nothing)
    nx, ny, nz = size(cache["ta"])
    dest_full = Array{Float32}(undef, nx, ny, nz)
    copyto!(dest_full, cache["ta"])

    z_hi = min(40, nz)
    idx = if nz >= 94
        vcat(collect(10:24), collect(80:94))
    else
        mid = max(1, nz ÷ 2)
        vcat(collect(1:min(5, mid)), collect(max(mid + 1, nz - 3):nz))
    end

    println("== benchmark_remote_zarr_slab_reads: site=$site month=$month t=$t1 experiment=$(repr(experiment)) ==")
    println("  field=T→ta  nz=$nz  contiguous_subset=1:$z_hi  vector_idx_len=$(length(idx))")

    dest_sub = Array{Float32}(undef, nx, ny, z_hi)
    dest_vec = Array{Float32}(undef, nx, ny, length(idx))

    t_full, s_full = _median_secs() do
        _materialize_ta!(cache, ds, t1, nothing, dest_full)
    end
    t_sub, s_sub = _median_secs() do
        _materialize_ta!(cache, ds, t1, 1:z_hi, dest_sub)
    end
    t_vec, s_vec = _median_secs() do
        _materialize_ta!(cache, ds, t1, idx, dest_vec)
    end

    _fmt_ms(t) = string(round(t * 1000; digits=2), " ms")
    println("\nMedian wall time — load + copyto! (dense), after warmup ($(N_SAMPLES) samples each):")
    println("  full_z (all levels)      median=$( _fmt_ms(t_full))  (min…max ms: $(round(Statistics.minimum(s_full)*1000;digits=2)) … $(round(Statistics.maximum(s_full)*1000;digits=2)))")
    println("  contiguous z 1:$z_hi      median=$( _fmt_ms(t_sub))  (min…max: $(round(Statistics.minimum(s_sub)*1000;digits=2)) … $(round(Statistics.maximum(s_sub)*1000;digits=2)))")
    println("  fused vector z (len=$(length(idx))) median=$( _fmt_ms(t_vec))  (min…max: $(round(Statistics.minimum(s_vec)*1000;digits=2)) … $(round(Statistics.maximum(s_vec)*1000;digits=2)))")
    println("\nNote: first timing batch can still reflect warm HTTP/Zarr cache from the probe above; restart Julia for a colder run.")
end

main()
