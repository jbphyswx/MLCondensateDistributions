"""
Visualization namespace for MLCondensateDistributions.

This module defines the public plotting API and the small amount of
non-plotting output-path logic shared by the CairoMakie extension and the
experiment scripts. All CairoMakie-dependent methods live in `ext/CairoMakieExt.jl`.
"""
module Viz

using Dates: Dates

export diagnostics_output_dir
export plot_targets_vs_resolution, plot_targets_heatmaps, plot_targets_binned_vs_x
export plot_training_diagnostics_from_artifact

"""Return the diagnostics output directory, using `latest` unless timestamped output is requested."""
function diagnostics_output_dir(base_dir::String; timestamped::Bool=false, stamp_fn::Function=() -> Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))
    if timestamped
        out_dir = joinpath(base_dir, "diagnostics_full", stamp_fn())
        mkpath(out_dir)
        return out_dir
    end

    out_dir = joinpath(base_dir, "diagnostics_full", "latest")
    if isdir(out_dir)
        rm(out_dir; recursive=true, force=true)
    end
    mkpath(out_dir)
    return out_dir
end

"""Plot one panel per target versus a resolution axis. Requires CairoMakie via the package extension."""
function plot_targets_vs_resolution(args...; kwargs...)
    error("CairoMakie is required for plot_targets_vs_resolution")
end

"""Plot resolution heatmaps for target columns. Requires CairoMakie via the package extension."""
function plot_targets_heatmaps(args...; kwargs...)
    error("CairoMakie is required for plot_targets_heatmaps")
end

"""Plot binned median of each target vs a continuous x column. Requires CairoMakie via the package extension."""
function plot_targets_binned_vs_x(args...; kwargs...)
    error("CairoMakie is required for plot_targets_binned_vs_x")
end

"""Render training diagnostics from a model artifact. Requires CairoMakie via the package extension."""
function plot_training_diagnostics_from_artifact(args...; kwargs...)
    error("CairoMakie is required for plot_training_diagnostics_from_artifact")
end

end # module