using Test: Test
using MLCondensateDistributions: MLCondensateDistributions as MLCD

Test.@testset "GoogleLES timestep profile accumulator" begin
    a = MLCD._GoogleLESProfAcc()
    Test.@test a.n == 0
    MLCD._googleles_prof_add!(a, 0.1, 0.02, 3.0, 0.5)
    Test.@test a.n == 1
    Test.@test a.qc_s ≈ 0.1
    Test.@test a.prep_s ≈ 0.02
    Test.@test a.nonqc_s ≈ 3.0
    Test.@test a.tabular_s ≈ 0.5
    s = MLCD._googleles_prof_fmt_avg(a)
    Test.@test occursin("n_cloudy=1", s)
    Test.@test occursin("nonqc_zarr=3.0", s)

    MLCD._googleles_prof_add!(a, 0.1, 0.02, 3.0, 0.5)
    Test.@test a.n == 2
    Test.@test occursin("n_cloudy=2", MLCD._googleles_prof_fmt_avg(a))
end
