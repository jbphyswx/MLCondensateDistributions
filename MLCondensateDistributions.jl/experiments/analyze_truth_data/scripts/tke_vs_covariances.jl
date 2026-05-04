#=

    Binned median of each covariance vs subgrid TKE.

        julia --project=experiments/analyze_truth_data experiments/analyze_truth_data/scripts/tke_vs_covariances.jl

    Optional stratification: one figure per distinct `resolution_h` (up to `max_strata`, sorted).

    Figures: experiments/analyze_truth_data/figures/tke_vs_covariances/

=#

include(joinpath(@__DIR__, "truth_analysis_common.jl"))

using DataFrames: DataFrames
using MLCondensateDistributions: MLCondensateDistributions as MLCD
using CairoMakie: CairoMakie

const FIG_DIR = joinpath(@__DIR__, "..", "figures", "tke_vs_covariances")

function run_tke_vs_covariances!(;
    data_dir::String = MLCD.Paths.processed_data_root(),
    max_files::Int = 0,
    data_source::Union{Nothing,String} = nothing,
    forcing_model::Union{Nothing,String} = nothing,
    experiment::Union{Nothing,String} = nothing,
    stratify_resolution_h::Bool = false,
    max_strata::Int = 8,
    nbins::Int = 40,
)
    extra = [:tke, :data_source, :forcing_model, :experiment]
    @info "Loading processed Arrow data for TKE vs covariances" data_dir
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
        :tke,
        joinpath(FIG_DIR, "cov_vs_tke.png");
        nbins=nbins,
        title_suffix=suffix,
    )

    if stratify_resolution_h && DataFrames.nrow(df) > 0
        subdir = joinpath(FIG_DIR, "by_resolution_h")
        mkpath(subdir)
        hs = sort(unique(df[!, :resolution_h]))
        if length(hs) > max_strata
            @warn "Many resolution_h values; plotting first max_strata only" n=length(hs) max_strata
            hs = hs[1:max_strata]
        end
        for h in hs
            sub = DataFrames.subset(df, :resolution_h => DataFrames.ByRow(==(h)))
            if DataFrames.nrow(sub) < 100
                @info "Skipping sparse stratum" resolution_h=h rows=DataFrames.nrow(sub)
                continue
            end
            tag = replace(string(Float64(h)), '.' => 'p')
            MLCD.Viz.plot_targets_binned_vs_x(
                sub,
                covs,
                :tke,
                joinpath(subdir, "cov_vs_tke_resolution_h_$(tag).png");
                nbins=nbins,
                title_suffix=" (resolution_h=$(h))",
            )
        end
    end

    @info "Saved figures" figure_dir=FIG_DIR
    return FIG_DIR
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tke_vs_covariances!()
end
