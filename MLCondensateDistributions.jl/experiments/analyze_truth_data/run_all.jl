#=

    Run all truth-analysis figure scripts with shared options.

        julia --project=experiments/analyze_truth_data experiments/analyze_truth_data/run_all.jl

    From Julia: `include("experiments/analyze_truth_data/run_all.jl"); run_all_truth_analyses!(; max_files=3)`

=#

using MLCondensateDistributions: MLCondensateDistributions as MLCD

include(joinpath(@__DIR__, "scripts", "resolution_impact_covariances.jl"))
include(joinpath(@__DIR__, "scripts", "tke_vs_covariances.jl"))
include(joinpath(@__DIR__, "scripts", "fractions_vs_covariances.jl"))

function run_all_truth_analyses!(;
    data_dir::String = MLCD.Paths.processed_data_root(),
    max_files::Int = 0,
    data_source::Union{Nothing,String} = nothing,
    forcing_model::Union{Nothing,String} = nothing,
    experiment::Union{Nothing,String} = nothing,
    stratify_resolution_h::Bool = false,
)
    run_resolution_impact_covariances!(; data_dir, max_files)
    run_tke_vs_covariances!(;
        data_dir,
        max_files,
        data_source,
        forcing_model,
        experiment,
        stratify_resolution_h,
    )
    run_fractions_vs_covariances!(;
        data_dir,
        max_files,
        data_source,
        forcing_model,
        experiment,
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_truth_analyses!()
end
