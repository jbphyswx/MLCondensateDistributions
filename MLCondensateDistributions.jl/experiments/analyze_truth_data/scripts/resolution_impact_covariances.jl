#=

    Here, we explore how vertical and horizontal GCM resolution (so domain size from LES averaging) impact our calculated covariances.

    Run from repo root (or this directory with adjusted paths):

        julia --project=experiments/analyze_truth_data experiments/analyze_truth_data/scripts/resolution_impact_covariances.jl

    Input: `Paths.processed_data_root()` → `<package>/data/processed/*.arrow`.
    Smoke test: `run_resolution_impact_covariances!(; max_files=3)`.

    Figures: experiments/analyze_truth_data/figures/resolution_impact_covariances/

=#

using DataFrames: DataFrames
using MLCondensateDistributions: MLCondensateDistributions as MLCD
using CairoMakie: CairoMakie as CM # Needed for the extension (activates CairoMakieExt; `MLCD.Viz` plot methods).

include(joinpath(@__DIR__, "truth_analysis_common.jl"))

const FIG_DIR = joinpath(@__DIR__, "..", "figures", "resolution_impact_covariances")

function run_resolution_impact_covariances!(; data_dir::String = MLCD.Paths.processed_data_root(), max_files::Int = 0)
    @info "Loading processed Arrow data for resolution-impact analysis" data_dir
    df, targets = MLCD.DataHandling.load_moments_dataframe(data_dir; max_files=max_files)
    @info "Loaded rows for analysis" rows=DataFrames.nrow(df) n_targets=length(targets)

    mkpath(FIG_DIR)

    cov_cols = covariance_targets(targets)
    corr_targets = append_pearson_correlation_columns!(df, cov_cols)
    @info "Pearson correlation columns" n=length(corr_targets) cols=corr_targets

    MLCD.Viz.plot_targets_vs_resolution(
        df,
        targets,
        :resolution_h,
        joinpath(FIG_DIR, "moments_vs_resolution_h.png"),
    )

    MLCD.Viz.plot_targets_vs_resolution(
        df,
        targets,
        :resolution_z,
        joinpath(FIG_DIR, "moments_vs_resolution_z.png"),
    )

    MLCD.Viz.plot_targets_heatmaps(
        df,
        targets,
        joinpath(FIG_DIR, "moments_resolution_heatmaps.png"),
    )

    if !isempty(corr_targets)
        MLCD.Viz.plot_targets_vs_resolution(
            df,
            corr_targets,
            :resolution_h,
            joinpath(FIG_DIR, "correlations_vs_resolution_h.png"),
        )

        MLCD.Viz.plot_targets_vs_resolution(
            df,
            corr_targets,
            :resolution_z,
            joinpath(FIG_DIR, "correlations_vs_resolution_z.png"),
        )

        MLCD.Viz.plot_targets_heatmaps(
            df,
            corr_targets,
            joinpath(FIG_DIR, "correlations_resolution_heatmaps.png"),
        )
    else
        @warn "No corr_* columns were added (missing cov_* / var_* pairs?); skipping correlation figures."
    end

    @info "Saved figures" figure_dir=FIG_DIR
    return FIG_DIR
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_resolution_impact_covariances!()
end
