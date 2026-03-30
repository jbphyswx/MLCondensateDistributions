using Test: Test

Test.@testset "NN_Condensate_Distributions Full Test Suite" begin
    include("test_cfSites.jl")
    include("test_googleles.jl")
    include("test_dataset_builder.jl")
    include("test_full_pipeline.jl")
end

