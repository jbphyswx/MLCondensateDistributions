using Test: Test
using MLCondensateDistributions: MLCondensateDistributions as MLCD

Test.@testset "GoogleLES non-qc load strategy (nonqc_strategy)" begin
    # Arguments (n_spans, n_keep, nz) are reserved for future policy; behavior is strategy-only today.
    args = (3, 120, 480)

    Test.@test !MLCD._googleles_use_full_nonqc_timestep_load(args...; nonqc_strategy="auto")

    for mode in ("auto", "per_span", "sparse", "minimal", "AuTo", "  per_span  ")
        Test.@test !MLCD._googleles_use_full_nonqc_timestep_load(args...; nonqc_strategy=mode)
    end

    for mode in ("full", "full_timestep", "FULL", "  full_timestep  ")
        Test.@test MLCD._googleles_use_full_nonqc_timestep_load(args...; nonqc_strategy=mode)
    end

    Test.@test_throws ArgumentError MLCD._googleles_use_full_nonqc_timestep_load(
        args...;
        nonqc_strategy = "bogus_mode_xyz",
    )
end
