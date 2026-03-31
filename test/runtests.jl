using Test: Test

Test.@testset "MLCondensateDistributions Full Test Suite" begin
    include("test_cfSites.jl")
    include("test_googleles.jl")
    include("test_dataset_builder.jl")
    include("test_full_pipeline.jl")
end

