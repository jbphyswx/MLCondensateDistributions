#!/usr/bin/env julia
# A/B wall time: TabularBuildOptions(nonqc_strategy="auto") vs "full_timestep".
# Uses timestep_profile=true so you see running avg nonqc_zarr in the log.
#
# Not comparable to `benchmark_remote_zarr_slab_reads.jl`: that script times one field / few `copyto!` patterns;
# this script times **entire** `build_tabular` (all fields, `nt` timesteps, tabular CPU, Arrow, etc.).
#
# The printed **SUMMARY** is `time()` around the whole `build_tabular` — not the profile’s `nonqc_zarr` average.
# To compare non-QC Zarr seconds, read the timestep profile lines / final “profile summary” in the log (`nonqc_zarr=...`).
#
# Runs **two** orderings in one invocation (auto→full, then full→auto) so warm-cache bias is visible on both sides.
# Compare `nonqc_zarr` in the logs; end-to-end wall still includes Arrow, tabular CPU, etc.
#
# Usage (from repo root):
#   julia --project=. experiments/amip_baseline/profile/benchmark_nonqc_strategy.jl [site] [month] [nt] [experiment]
#
# Do not pipe stdout to `head`/`tail` while the job runs: after the pipe closes, Julia can block
# on the next write (full pipe buffer). Tee to a file is fine: `julia ... 2>&1 | tee bench.log`

const ROOT = dirname(dirname(dirname(@__DIR__)))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD

function main()
    site = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 313
    month = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    nt = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
    experiment = length(ARGS) >= 4 ? ARGS[4] : "amip"
    println(stderr, "benchmark_nonqc_strategy: after q_c cache, each timestep may take minutes on WAN; progress uses [progress] lines when stdout is not a TTY.")
    println("== benchmark_nonqc_strategy: site=$site month=$month nt=$nt experiment=$(repr(experiment)) ==")
    flush(stderr)
    flush(stdout)
    function _run(label::String, nonqc::String)
        opts = MLCD.TabularBuildOptions(nonqc_strategy=nonqc, timestep_profile=true)
        out = mktempdir(prefix="mlcd_nonqc_ab_")
        t0 = time()
        MLCD.GoogleLES.build_tabular(site, month, experiment, out; max_timesteps=nt, verbose=true, tabular_options=opts)
        wall = time() - t0
        println("\n>>> RESULT $label  wall_seconds=$(round(wall; digits=2))  out=$out\n")
        return (label, wall, out)
    end

    results = Tuple{String,Float64,String}[]
    for (label, nonqc) in (("NONQC_auto_per_span", "auto"), ("NONQC_full_timestep", "full_timestep"))
        push!(results, _run(label, nonqc))
    end
    println("=== A/B SUMMARY order=auto_then_full (end-to-end wall time) ===")
    for (label, wall, _) in results
        println("  $label  $(round(wall; digits=3)) s")
    end
    if length(results) == 2
        w1, w2 = results[1][2], results[2][2]
        println("  ratio full_timestep/auto = $(round(w2 / w1; digits=3))  (values >1 mean full_timestep slower)")
        println("  ratio auto/full_timestep = $(round(w1 / w2; digits=3))  (values >1 mean per-span slower)")
    end

    println("\n--- Second pass: full_timestep first, then auto (isolates warm-cache bias on one ordering) ---")
    results2 = Tuple{String,Float64,String}[]
    for (label, nonqc) in (("NONQC_full_timestep_first", "full_timestep"), ("NONQC_auto_second", "auto"))
        push!(results2, _run(label, nonqc))
    end
    println("=== A/B SUMMARY order=full_then_auto ===")
    for (label, wall, _) in results2
        println("  $label  $(round(wall; digits=3)) s")
    end
    if length(results2) == 2
        wf, wa = results2[1][2], results2[2][2]
        println("  ratio auto/full (this pass) = $(round(wa / wf; digits=3))")
    end
end

main()
