using Test: Test

import MLCondensateDistributions as MLCD

include("../utils/reduction_specs.jl")
using .ReductionSpecs

Test.@testset "ReductionSpecs truncated extent (124 × nh=21)" begin
    n_used, disc = truncated_block_extent(124, 21)
    Test.@test n_used == 105
    Test.@test disc == 19
end

Test.@testset "ReductionSpecs truncated_horizontal_sizes legacy flat subsample vs full band" begin
    # `truncated_horizontal_sizes` remains a flat [nh_min, nh_max] subsampler (research / helpers).
    nhs124 = truncated_horizontal_sizes(124, 124, 50.0f0, 1000.0f0)
    Test.@test nhs124 == subsample_closed_range(20, 124, 5)
    Test.@test !(21 in nhs124)
    nhs124full = truncated_horizontal_sizes(124, 124, 50.0f0, 1000.0f0; horizontal_budget = nothing)
    Test.@test nhs124full == collect(20:124)
    Test.@test 21 in nhs124full
    nhs120full = truncated_horizontal_sizes(120, 120, 50.0f0, 1000.0f0; horizontal_budget = nothing)
    Test.@test nhs120full == collect(20:120)
end

Test.@testset "ReductionSpecs horizontal_sliding_budget_split" begin
    Test.@test horizontal_sliding_budget_split(10) == (5, 5)
    Test.@test horizontal_sliding_budget_split(9) == (4, 5)
    Test.@test horizontal_sliding_budget_split(1) == (1, 0)
end

Test.@testset "ReductionSpecs horizontal_block_factors_from_tower (§3 seeds + prime DFS)" begin
    nhs = horizontal_block_factors_from_tower(124, 124, 50.0f0, 1000.0f0; horizontal_budget = 10)
    Test.@test issorted(nhs)
    Test.@test all(nh -> nh <= 124 && nh >= 1, nhs)
    Test.@test 20 in nhs
    Test.@test 124 in nhs
    # Seeds live in [nh_min, nh_half] = [20, 62]; tower may reach nh > 62 (§3.3).
    nh_half = 62
    nh_min = 20
    seeds = subsample_closed_range(nh_min, nh_half, 5)
    Test.@test all(s -> s in nhs, seeds)
end

Test.@testset "ReductionSpecs block_reduction_triples uses horizontal tower + vertical tower" begin
    nx, ny, nz = 124, 124, 8
    dx = 50.0f0
    min_h = MLCD.GoogleLES.GoogleLESDefaults.min_horizontal_averaging_domain
    dz_ref = 20.0f0
    max_dz = MLCD.GoogleLES.GoogleLESDefaults.max_vertical_averaging_domain
    triples = block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz)
    Test.@test !isempty(triples)
    Test.@test any(t -> t[1] == 124 && t[2] == 124, triples)
    Test.@test any(t -> t[1] == 20 && t[2] == 20, triples)
    nhs = unique!(sort!([t[1] for t in triples]))
    block_nhs = Set(nhs)
    tower_nhs = horizontal_block_factors_from_tower(nx, ny, dx, min_h; horizontal_budget = 10)
    Test.@test Set(tower_nhs) == block_nhs
end

Test.@testset "ReductionSpecs vertical tower includes non-divisors of nz (prime nz)" begin
    nz = 7
    dz_ref = 10.0f0
    max_dz = MLCD.GoogleLES.GoogleLESDefaults.max_vertical_averaging_domain
    Test.@test 3 in vertical_block_factors_from_tower(nz, dz_ref, max_dz; vertical_budget = 5)
    Test.@test effective_fz_max(nz, dz_ref, max_dz) == 7
end

Test.@testset "ReductionSpecs subsample_closed_range" begin
    s = subsample_closed_range(20, 124, 5)
    Test.@test length(s) == 5
    Test.@test s[1] == 20
    Test.@test s[end] == 124
    Test.@test issorted(s)
    Test.@test subsample_closed_range(5, 5, 10) == [5]
end

Test.@testset "ReductionSpecs sliding_reduction_triples budgets horizontal sizes" begin
    nx, ny, nz = 124, 124, 8
    dx = 50.0f0
    min_h = 1000.0f0
    dz_ref = 20.0f0
    max_dz = 400.0f0
    trip5 = sliding_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz; horizontal_budget = 5)
    nhs5 = unique!(sort!([t[1] for t in trip5]))
    Test.@test length(nhs5) <= 5
    trip_full = block_reduction_triples(
        nx,
        ny,
        nz,
        dx,
        min_h,
        dz_ref,
        max_dz;
        horizontal_budget = nothing,
    )
    nhs_full = unique!(sort!([t[1] for t in trip_full]))
    Test.@test length(nhs5) <= length(nhs_full)
end

Test.@testset "ReductionSpecs hybrid_sliding_extra_sizes_default upper band only (§4)" begin
    nx, ny = 124, 124
    dx = 50.0f0
    min_h = 1000.0f0
    nz = 8
    dz_ref = 20.0f0
    max_dz = 400.0f0
    triples = block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz; horizontal_budget = 10)
    block_nhs = Set(t[1] for t in triples)
    _, b_hi = horizontal_sliding_budget_split(10)
    extra = hybrid_sliding_extra_sizes_default(nx, ny, dx, min_h, block_nhs; budget = b_hi)
    nh_half = fld(min(nx, ny), 2)
    nh_min = max(1, ceil(Int, min_h / dx))
    lo = max(nh_min, nh_half)
    Test.@test !isempty(extra)
    Test.@test all(w -> w >= lo && w <= min(nx, ny) && !(w in block_nhs), extra)
end

Test.@testset "ReductionSpecs hybrid_sliding_extra_sizes_default skips sparse block nh (upper band)" begin
    block_nhs = Set([20, 40, 60, 80, 100, 120, 124])
    extra = hybrid_sliding_extra_sizes_default(124, 124, 50.0f0, 1000.0f0, block_nhs; budget = 5)
    Test.@test all(w -> !(w in block_nhs), extra)
    Test.@test all(w -> w >= 62, extra)  # nh_half=62, nh_slide_lo = max(20,62)
    Test.@test !isempty(extra)
end

Test.@testset "conv3d_block_mean 124×124×1 with fx=21 yields 5×5" begin
    if !isdefined(Main, :ArrayUtils)
        include(joinpath(@__DIR__, "..", "utils", "array_utils.jl"))
    end
    using .ArrayUtils: ArrayUtils
    a = rand(Float32, 124, 124, 1)
    b = ArrayUtils.conv3d_block_mean(a, 21, 21, 1)
    Test.@test size(b) == (5, 5, 1)
end
