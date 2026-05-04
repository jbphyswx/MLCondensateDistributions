"""
Compare old vs v2 GoogleLES build_tabular on real pipeline I/O + processing.

Usage:
    julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions \
      /home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile/profile_v2_pipeline_compare.jl

Optional env vars:
    SITE_ID=320
    MONTH=1
    EXPERIMENT=amip
    MAX_TIMESTEPS=4
"""

using Dates

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions

const SITE_ID = parse(Int, get(ENV, "SITE_ID", "320"))
const MONTH = parse(Int, get(ENV, "MONTH", "1"))
const EXPERIMENT = get(ENV, "EXPERIMENT", "amip")
const MAX_TIMESTEPS = parse(Int, get(ENV, "MAX_TIMESTEPS", "4"))

const OUT_OLD = mktempdir()
const OUT_V2 = mktempdir()

function time_call(label, f)
    t0 = time()
    f()
    dt = time() - t0
    println(label, " seconds=", round(dt; digits=3))
    return dt
end

println("== profile_v2_pipeline_compare ==")
println("site=", SITE_ID, " month=", MONTH, " experiment=", EXPERIMENT, " max_timesteps=", MAX_TIMESTEPS)
println("out_old=", OUT_OLD)
println("out_v2=", OUT_V2)

# Warmup (small) so we compare more stable runtime behavior.
MLCondensateDistributions.GoogleLES.build_tabular(SITE_ID, MONTH, EXPERIMENT, OUT_OLD; max_timesteps=1, verbose=false)
MLCondensateDistributions.GoogleLES.build_tabular_v2(SITE_ID, MONTH, EXPERIMENT, OUT_V2; max_timesteps=1, verbose=false)

old_s = time_call("old_build_tabular", () ->
    MLCondensateDistributions.GoogleLES.build_tabular(
        SITE_ID,
        MONTH,
        EXPERIMENT,
        OUT_OLD;
        max_timesteps=MAX_TIMESTEPS,
        verbose=false,
    )
)

v2_s = time_call("v2_build_tabular", () ->
    MLCondensateDistributions.GoogleLES.build_tabular_v2(
        SITE_ID,
        MONTH,
        EXPERIMENT,
        OUT_V2;
        max_timesteps=MAX_TIMESTEPS,
        verbose=false,
    )
)

println("speed_ratio_old_over_v2=", round(old_s / v2_s; digits=3))
