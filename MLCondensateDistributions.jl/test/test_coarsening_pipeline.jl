using Test: Test
using Random: Random
using Statistics: Statistics

if !isdefined(Main, :ArrayUtils)
    include(joinpath(@__DIR__, "..", "utils", "array_utils.jl"))
end
using .ArrayUtils: ArrayUtils
include(joinpath(@__DIR__, "..", "utils", "statistical_methods", "StatisticalMethods.jl"))
using .StatisticalMethods: StatisticalMethods
include("../utils/coarsening_pipeline.jl")
using .CoarseningPipeline: CoarseningPipeline

Test.@testset "CoarseningPipeline arbitrary seeds" begin
    # Test with seeds (1, 31) and no min_h constraint so we get the full ladder
    levels = CoarseningPipeline.build_horizontal_levels(124, 124, Float32(6000 / 123); seeds=(1, 31), min_h=Float32(0), include_full_domain=true)
    
    factors = [l.fx for l in levels]
    # seed 1: 1, 2, 4, 8, 16, 32, 64, 124
    # seed 31: 31, 62, 124
    Test.@test 1 in factors
    Test.@test 31 in factors
    Test.@test 62 in factors
    Test.@test 124 in factors
    Test.@test issorted(factors)
end

Test.@testset "CoarseningPipeline field + product coarsening" begin
    a = reshape(Float32.(1:32), 4, 4, 2)
    b = fill(Float32(2), 4, 4, 2)
    fields = Dict("a" => a, "b" => b)

    product_pairs = Dict(
        "aa" => ("a", "a"),
        "ab" => ("a", "b"),
    )

    views = CoarseningPipeline.build_horizontal_multilevel_views(fields, Float32(1); seeds=(1,), min_h=Float32(2), include_full_domain=true, product_pairs=product_pairs)
    Test.@test !isempty(views)

    lvl2 = only(filter(v -> v.fx == 2 && v.fy == 2, views))
    Test.@test size(lvl2.means["a"]) == (2, 2, 2)
    Test.@test size(lvl2.products["aa"]) == (2, 2, 2)

    # Covariance sanity check: cov(a,b) with b=constant should be zero.
    cov_ab = CoarseningPipeline.covariance_from_moments(lvl2.products["ab"], lvl2.means["a"], lvl2.means["b"])
    Test.@test Statistics.maximum(abs, cov_ab) ≈ 0f0 atol=1f-6
end

Test.@testset "CoarseningPipeline type stability" begin
    a = rand(Random.MersenneTwister(10), Float32, 8, 8, 4)
    b = rand(Random.MersenneTwister(11), Float32, 8, 8, 4)

    fields = Dict("a" => a, "b" => b)
    coarse = Test.@inferred CoarseningPipeline.coarsen_fields_at_level(fields, 2, 2)
    Test.@test size(coarse["a"]) == (4, 4, 4)

    products = Dict("ab" => ("a", "b"))
    prod = Test.@inferred CoarseningPipeline.coarsen_products_at_level(fields, products, 2, 2)
    Test.@test size(prod["ab"]) == (4, 4, 4)
end

Test.@testset "CoarseningPipeline convolutional triples and 3D products" begin
    triples = CoarseningPipeline.build_convolutional_coarsening_triples(6, 6, 6, Float32(1), Float32(0), Float32(10), Float32(100))
    Test.@test (2, 2, 2) in triples
    Test.@test (3, 3, 3) in triples
    Test.@test (1, 1, 1) in triples

    a = reshape(Float32.(1:(4 * 4 * 4)), 4, 4, 4)
    b = fill(Float32(2), 4, 4, 4)
    fields = (hus = a, thetali = b)
    pairs = (qt_h = (:hus, :thetali),)
    p3 = CoarseningPipeline.coarsen_products_3d_block(fields, pairs, 2, 2, 2)
    m1 = ArrayUtils.conv3d_block_mean(a, 2, 2, 2)
    m2 = ArrayUtils.conv3d_block_mean(b, 2, 2, 2)
    Test.@test p3.qt_h ≈ m1 .* m2
end

Test.@testset "CoarseningPipeline allocation regression" begin
    dx = Float32(50)
    min_h = Float32(1000)

    a = rand(Random.MersenneTwister(42), Float32, 124, 124, 120)
    b = rand(Random.MersenneTwister(43), Float32, 124, 124, 120)

    fields_concrete = Dict{String, Array{Float32, 3}}("a" => a, "b" => b)
    fields_abstract = Dict{String, AbstractArray{Float32, 3}}("a" => a, "b" => b)
    product_pairs = Dict("ab" => ("a", "b"))

    _ = CoarseningPipeline.build_horizontal_multilevel_views(fields_concrete, dx; seeds=(1,), min_h=min_h, include_full_domain=false, product_pairs=product_pairs)
    _ = CoarseningPipeline.build_horizontal_multilevel_views(fields_abstract, dx; seeds=(1,), min_h=min_h, include_full_domain=false, product_pairs=product_pairs)

    alloc_concrete = @allocated CoarseningPipeline.build_horizontal_multilevel_views(fields_concrete, dx; seeds=(1,), min_h=min_h, include_full_domain=false, product_pairs=product_pairs)
    alloc_abstract = @allocated CoarseningPipeline.build_horizontal_multilevel_views(fields_abstract, dx; seeds=(1,), min_h=min_h, include_full_domain=false, product_pairs=product_pairs)

    # Concrete dictionaries should remain in the low-MB range for this benchmark shape.
    Test.@test alloc_concrete < 2_000_000

    # Guard against accidentally reintroducing abstract-container dispatch in hot paths.
    Test.@test alloc_abstract > 100 * alloc_concrete
end
