#=

    Binned median of covariances vs `liq_fraction` and vs `ice_fraction`.

        julia --project=experiments/analyze_truth_data experiments/analyze_truth_data/scripts/fractions_vs_covariances.jl

    Figures: experiments/analyze_truth_data/figures/fractions_vs_covariances/

=#

include(joinpath(@__DIR__, "truth_analysis_common.jl"))

using DataFrames: DataFrames
using MLCondensateDistributions: MLCondensateDistributions as MLCD
using CairoMakie: CairoMakie

const FIG_DIR = joinpath(@__DIR__, "..", "figures", "fractions_vs_covariances")

function run_fractions_vs_covariances!(;
    data_dir::String = MLCD.Paths.processed_data_root(),
    max_files::Int = 0,
    data_source::Union{Nothing,String} = nothing,
    forcing_model::Union{Nothing,String} = nothing,
    experiment::Union{Nothing,String} = nothing,
    nbins::Int = 40,
)
    extra = [:liq_fraction, :ice_fraction, :data_source, :forcing_model, :experiment]
    @info "Loading processed Arrow data for fraction vs covariances" data_dir
    df, targets = MLCD.DataHandling.load_moments_dataframe(
        data_dir;
        max_files=max_files,
        target_prefixes=["cov_"],
        extra_columns=extra,
    )
    df = apply_provenance_filters(df; data_source, forcing_model, experiment)
    covs = covariance_targets(targets)
    @info "Rows after provenance filter" rows=DataFrames.nrow(df) n_covariances=length(covs)

    mkpath(FIG_DIR)
    suffix = ""
    if data_source !== nothing || forcing_model !== nothing || experiment !== nothing
        suffix = " (filtered)"
    end

    MLCD.Viz.plot_targets_binned_vs_x(
        df,
        covs,
        :liq_fraction,
        joinpath(FIG_DIR, "cov_vs_liq_fraction.png");
        nbins=nbins,
        title_suffix=suffix,
    )

    MLCD.Viz.plot_targets_binned_vs_x(
        df,
        covs,
        :ice_fraction,
        joinpath(FIG_DIR, "cov_vs_ice_fraction.png");
        nbins=nbins,
        title_suffix=suffix,
    )

    @info "Saved figures" figure_dir=FIG_DIR
    return FIG_DIR
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fractions_vs_covariances!()
end
