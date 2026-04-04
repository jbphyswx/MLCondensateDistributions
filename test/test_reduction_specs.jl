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

Test.@testset "conv3d_block_mean 124×124×1 with fx=21 yields 5×5" begin
    include("../utils/array_utils.jl")
    using .ArrayUtils: conv3d_block_mean
    a = rand(Float32, 124, 124, 1)
    b = conv3d_block_mean(a, 21, 21, 1)
    Test.@test size(b) == (5, 5, 1)
end
