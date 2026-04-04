using Test: Test
using MLCondensateDistributions: MLCondensateDistributions as M

Test.@testset "GoogleLES non-qc load strategy (MLCD_GOOGLELES_NONQC_STRATEGY)" begin
    # Arguments (n_spans, n_keep, nz) are reserved for future policy; behavior is env-only today.
    args = (3, 120, 480)

    Test.withenv("MLCD_GOOGLELES_NONQC_STRATEGY" => nothing) do
        Test.@test !M._googleles_use_full_nonqc_timestep_load(args...)
    end

    for mode in ("auto", "per_span", "sparse", "minimal", "AuTo", "  per_span  ")
        Test.withenv("MLCD_GOOGLELES_NONQC_STRATEGY" => mode) do
            Test.@test !M._googleles_use_full_nonqc_timestep_load(args...)
        end
    end

    for mode in ("full", "full_timestep", "FULL", "  full_timestep  ")
        Test.withenv("MLCD_GOOGLELES_NONQC_STRATEGY" => mode) do
            Test.@test M._googleles_use_full_nonqc_timestep_load(args...)
        end
    end

    Test.withenv("MLCD_GOOGLELES_NONQC_STRATEGY" => "bogus_mode_xyz") do
        Test.@test !M._googleles_use_full_nonqc_timestep_load(args...)
    end
end
