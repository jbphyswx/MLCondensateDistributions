using Test: Test

include("../utils/reduction_specs.jl")
using .ReductionSpecs

Test.@testset "ReductionSpecs truncated extent (124 × nh=21)" begin
    n_used, disc = truncated_block_extent(124, 21)
    Test.@test n_used == 105
    Test.@test disc == 19
end

Test.@testset "ReductionSpecs block_reduction_triples uses truncated tiling counts" begin
    nx, ny, nz = 124, 124, 8
    dx = 50.0f0
    min_h = 1000.0f0
    dz_ref = 20.0f0
    max_dz = 400.0f0
    triples = block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz)
    Test.@test !isempty(triples)
    has_21 = any(t -> t[1] == 21 && t[2] == 21, triples)
    Test.@test has_21
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
    trip_full = block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz)
    nhs_full = unique!(sort!([t[1] for t in trip_full]))
    Test.@test length(nhs5) <= length(nhs_full)
end

Test.@testset "ReductionSpecs hybrid_sliding_extra_sizes_default skips block ladder" begin
    nx, ny = 124, 124
    dx = 50.0f0
    min_h = 1000.0f0
    nz = 8
    dz_ref = 20.0f0
    max_dz = 400.0f0
    triples = block_reduction_triples(nx, ny, nz, dx, min_h, dz_ref, max_dz)
    block_nhs = Set(t[1] for t in triples)
    extra = hybrid_sliding_extra_sizes_default(nx, ny, dx, min_h, block_nhs; budget = 5)
    Test.@test all(w -> !(w in block_nhs), extra)
end

Test.@testset "conv3d_block_mean 124×124×1 with fx=21 yields 5×5" begin
    include("../utils/array_utils.jl")
    using .ArrayUtils: conv3d_block_mean
    a = rand(Float32, 124, 124, 1)
    b = conv3d_block_mean(a, 21, 21, 1)
    Test.@test size(b) == (5, 5, 1)
end
