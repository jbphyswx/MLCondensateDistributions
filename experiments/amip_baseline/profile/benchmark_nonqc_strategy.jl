#!/usr/bin/env julia
# A/B wall time: MLCD_GOOGLELES_NONQC_STRATEGY=auto (per-span) vs full_timestep.
# Also sets MLCD_GOOGLELES_TIMESTEP_PROFILE=1 so you see running avg nonqc_zarr in the log.
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
    results = Tuple{String,Float64,String}[]
    for (label, nonqc) in (("NONQC_auto_per_span", "auto"), ("NONQC_full_timestep", "full_timestep"))
        wall, out = withenv(
            "MLCD_GOOGLELES_NONQC_STRATEGY" => nonqc,
            "MLCD_GOOGLELES_TIMESTEP_PROFILE" => "1",
        ) do
            out = mktempdir(prefix="mlcd_nonqc_ab_")
            t0 = time()
            MLCD.GoogleLES.build_tabular(site, month, experiment, out; max_timesteps=nt, verbose=true)
            time() - t0, out
        end
        println("\n>>> RESULT $label  wall_seconds=$(round(wall; digits=2))  out=$out\n")
        push!(results, (label, wall, out))
    end
    println("=== A/B SUMMARY (end-to-end build_tabular wall time) ===")
    for (label, wall, _) in results
        println("  $label  $(round(wall; digits=3)) s")
    end
    if length(results) == 2
        w1, w2 = results[1][2], results[2][2]
        println("  ratio full_timestep/auto = $(round(w2 / w1; digits=3))  (values >1 mean full_timestep slower)")
        println("  ratio auto/full_timestep = $(round(w1 / w2; digits=3))  (values >1 mean per-span slower)")
    end
    println("Note: second run may be faster from warm HTTP/Zarr cache; swap loop order or repeat for a fair read.")
end

main()
