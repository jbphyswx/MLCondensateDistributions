using Test: Test

Test.@testset "MLCondensateDistributions Full Test Suite" begin
    include("test_dynamics.jl")
    include("test_dataset_builder.jl")
    include("test_dataset_builder_impl.jl")
    include("test_build_training_data.jl")
    include("test_data_hygiene.jl")
    include("test_workflow_state.jl")
    include("test_vertical_coarsening.jl")
    include("test_array_utils.jl")
    include("test_coarsening_pipeline.jl")
    include("test_googleles_z_chunk_grouping.jl")
    include("test_googleles_nonqc_strategy.jl")
    include("test_googleles_timestep_profile.jl")

    run_live = lowercase(strip(get(ENV, "MLCD_RUN_LIVE_TESTS", "0"))) in ("1", "true", "yes", "y", "on")
    if run_live
        include("test_cfsites.jl")
        include("test_googleles.jl")
        include("test_full_pipeline.jl")
    else
        @info "Skipping live integration tests (set MLCD_RUN_LIVE_TESTS=1 to enable)."
    end
end

