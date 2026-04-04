using Test: Test
using Statistics: Statistics

include("../utils/array_utils.jl")
using .ArrayUtils

Test.@testset "ArrayUtils schedule from seeds" begin
    dx = Float32(6000 / 123)
    # Test with seeds (1, 31): generates chains 1*2^k and 31*2^k
    sched = build_schedule_from_seeds(124, dx; seeds=(1, 31), min_h=Float32(1000), include_full_domain=true)
    factors = getindex.(sched, :factor)
    Test.@test 31 in factors
    Test.@test 62 in factors
    Test.@test 124 in factors
    Test.@test issorted(factors)
end

Test.@testset "ArrayUtils seed_factor_ladder zero allocations" begin
    # Warmup
    _ = seed_factor_ladder(124, 1; min_factor=1)
    
    # Test that seed_factor_ladder allocates zero in hot path
    allocs = @allocations seed_factor_ladder(124, 1; min_factor=1)
    Test.@test allocs == 0
    
    allocs = @allocations seed_factor_ladder(124, 31; min_factor=1)
    Test.@test allocs == 0
end

Test.@testset "ArrayUtils build_schedule_from_seeds! zero allocations" begin
    dx = Float32(6000 / 123)
    schedule_out = ArrayUtils.ScheduleRow{Float32}[]
    factors_tmp = Int[]
    seeds = (1, 31)
    min_factor = ceil(Int, Float32(1000) / dx)

    _ = build_schedule_from_seeds!(
        schedule_out,
        factors_tmp,
        124,
        dx,
        seeds,
        min_factor,
        true,
    )

    allocs = @allocations build_schedule_from_seeds!(
        schedule_out,
        factors_tmp,
        124,
        dx,
        seeds,
        min_factor,
        true,
    )
    Test.@test allocs == 0
end

Test.@testset "ArrayUtils 2D mean pooling" begin
    a = reshape(Float32.(1:16), 4, 4)
    out = coarsen2d_mean(a, 2, 2)
    Test.@test size(out) == (2, 2)

    # Expected values from block means:
    # [ (1+2+5+6)/4   (9+10+13+14)/4
    #   (3+4+7+8)/4   (11+12+15+16)/4 ]
    Test.@test out[1, 1] ≈ Float32(3.5)
    Test.@test out[2, 1] ≈ Float32(5.5)
    Test.@test out[1, 2] ≈ Float32(11.5)
    Test.@test out[2, 2] ≈ Float32(13.5)
end

Test.@testset "ArrayUtils 3D pooling and full-domain" begin
    a = reshape(Float32.(1:32), 4, 4, 2)
    out = coarsen3d_horizontal_mean(a, 2, 2)
    Test.@test size(out) == (2, 2, 2)

    fd = full_domain_mean_3d(a)
    Test.@test length(fd) == 2
    Test.@test fd[1] ≈ Statistics.mean(@view a[:, :, 1])
    Test.@test fd[2] ≈ Statistics.mean(@view a[:, :, 2])
end

Test.@testset "ArrayUtils coarsen_dz_profile_factor" begin
    dz = Float32[1, 2, 3, 4, 5, 6]
    Test.@test coarsen_dz_profile_factor(dz, 2) ≈ Float32[3, 7, 11]
    Test.@test coarsen_dz_profile_factor(dz, 3) ≈ Float32[6, 15]
    Test.@test coarsen_dz_profile_2x(dz) == coarsen_dz_profile_factor(dz, 2)
    dz_odd = Float32[1, 1, 1]
    Test.@test coarsen_dz_profile_2x(dz_odd) ≈ Float32[2]
end

Test.@testset "ArrayUtils conv3d_block_mean vs horizontal+vertical" begin
    data = reshape(Float32.(1:(4 * 4 * 4)), 4, 4, 4)
    j = conv3d_block_mean(data, 2, 2, 2)
    h = coarsen3d_horizontal_mean(data, 2, 2)
    v = coarsen3d_vertical_mean(h, 2)
    Test.@test size(j) == (2, 2, 2)
    Test.@test j ≈ v
end

Test.@testset "ArrayUtils 3D pooling in-place zero allocations" begin
    # Warmup
    a = reshape(Float32.(1:32), 4, 4, 2)
    out = similar(a, 2, 2, 2)
    _ = coarsen3d_horizontal_mean!(out, a, 2, 2)
    
    # Test in-place coarsening allocates zero
    allocs = @allocations coarsen3d_horizontal_mean!(out, a, 2, 2)
    Test.@test allocs == 0
end
