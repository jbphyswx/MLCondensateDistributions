#!/usr/bin/env julia
# Compares MLCD_GOOGLELES_Z_CHUNK_MERGE=0 vs 1 (hull read + selective copy vs one read per span).
#
# Usage (from repo root):
#   julia --project=. experiments/amip_baseline/profile/benchmark_z_chunk_merge.jl [site] [month] [nt]

const ROOT = dirname(dirname(dirname(@__DIR__)))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD

function main()
    site = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 313
    month = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    nt = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
    experiment = "amip"
    println("== benchmark_z_chunk_merge: site=$site month=$month nt=$nt ==")
    for (label, merge_val) in (("MERGE_OFF", "0"), ("MERGE_ON", "1"))
        ENV["MLCD_GOOGLELES_Z_CHUNK_MERGE"] = merge_val
        out = mktempdir(prefix="mlcd_merge_bench_")
        t0 = time()
        MLCD.GoogleLES.build_tabular(site, month, experiment, out; max_timesteps=nt, verbose=true)
        println("\n>>> RESULT $label  wall_seconds=$(round(time() - t0; digits=2))  out=$out\n")
    end
    println("Note: second run may be faster from warm HTTP/Zarr cache.")
end

main()
