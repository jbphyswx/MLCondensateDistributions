#=
Build a **real** GoogleLES case Parquet file via remote Zarr (same entrypoint as production:
`GoogleLES.build_tabular`). Prints the file path and min/max of `var_*` and `tke` after reading the
file back—useful to inspect moment numerics on actual LES output.

Requires network access to the GoogleLES Zarr store.

Usage (from repository root is fine; the script activates the project):

    julia scripts/write_googleles_arrow_sample.jl

Environment (optional):

    OUTPUT_DIR=./my_out          # default: <repo>/artifacts/googleles_sample
    GOOGLELES_SITE=10            # default: 10
    GOOGLELES_MONTH=1            # default: 1
    GOOGLELES_EXPERIMENT=amip    # default: amip
    MAX_TIMESTEPS=1              # default: 1 (increase for more rows / longer run)
=#
using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

import MLCondensateDistributions as MLCD
using Parquet2: Parquet2
using DataFrames: DataFrames
using Printf: Printf
using Statistics: Statistics

function googleles_case_parquet_path(site_id::Int, month::Int, experiment::String, output_dir::String)
    site_s = lpad(string(site_id), 3, '0')
    month_s = lpad(string(month), 2, '0')
    return joinpath(output_dir, "googleles_case__$(site_s)__month__$(month_s)__exp__$(experiment).parquet")
end

function main()
    site = parse(Int, get(ENV, "GOOGLELES_SITE", "10"))
    month = parse(Int, get(ENV, "GOOGLELES_MONTH", "1"))
    experiment = get(ENV, "GOOGLELES_EXPERIMENT", "amip")
    output_dir = get(ENV, "OUTPUT_DIR", joinpath(@__DIR__, "..", "artifacts", "googleles_sample"))
    max_timesteps = parse(Int, get(ENV, "MAX_TIMESTEPS", "1"))

    mkpath(output_dir)
    output_dir = abspath(output_dir)
    parquet_path = abspath(googleles_case_parquet_path(site, month, experiment, output_dir))

    println("Building GoogleLES tabular data…")
    println("  site=$site month=$month experiment=$(repr(experiment)) max_timesteps=$max_timesteps")
    println("  output_dir=$output_dir")
    # Close HTTP pools after the run so idle-connection tasks do not warn at process exit.
    opts = MLCD.TabularBuildOptions(close_http_pools=true)
    MLCD.GoogleLES.build_tabular(site, month, experiment, output_dir; max_timesteps, verbose=true, tabular_options=opts)

    isfile(parquet_path) || error("Expected Parquet file not found: $parquet_path")

    df = DataFrames.DataFrame(Parquet2.Dataset(parquet_path))
    n = DataFrames.nrow(df)

    println()
    println("Parquet file: ", parquet_path)
    println("Rows: ", n, "  Cols: ", DataFrames.ncol(df))
    println()

    if n == 0
        println("No trainable rows in this slice (cloud mask / timestep window). Try another site/month or MAX_TIMESTEPS.")
        return
    end

    println("Min / max (negative min on var_* or tke ⇒ spurious Float32 cancellation in emitted moments):")
    for col in [:var_qt, :var_ql, :var_qi, :var_w, :var_h, :tke]
        v = df[!, col]
        mn = Statistics.minimum(v)
        mx = Statistics.maximum(v)
        Printf.@printf("  %-10s  min=% .6g  max=% .6g\n", col, mn, mx)
    end
    println()
    println("First 5 rows (selected columns):")
    show(stdout, MIME("text/plain"), DataFrames.first(df[:, [:resolution_h, :resolution_z, :var_qt, :var_w, :tke]], 5))
    println()
    println()
    println("Read in Julia:")
    println("  using Parquet2, DataFrames")
    println("  df = DataFrame(Parquet2.Dataset($(repr(parquet_path))))")
end

main()
