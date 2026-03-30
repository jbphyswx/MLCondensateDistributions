using Test: Test
using Zarr: Zarr

# Include the GoogleLES module from utils
include("../utils/GoogleLES.jl")
using .GoogleLES: GoogleLES as GL

Test.@testset "Google Cloud Zarr Access Tests" begin
    @info "Testing lazy metadata load from Google Cloud..."
    
    # Test valid site, month, experiment
    ds = GL.load_zarr_simulation(0, 1, "amip")
    
    # 1. Structure Check
    Test.@test ds !== nothing
    Test.@test typeof(ds) <: Zarr.ZGroup
    
    # 2. Variable Check (These are common SwirlLM features)
    # The dataset is a ZGroup, we can check if it has the required data arrays.
    Test.@test haskey(ds.arrays, "p_ref")
    Test.@test haskey(ds.arrays, "T")
    Test.@test haskey(ds.arrays, "q_t")
    
    # 3. Size Check (The raw arrays should have dimensions)
    # For instance temperature shape is (t, x, y, z) ~ (1, 124, 124, 60) per chunk typically,
    # but let's just assert its rank > 0
    t_arr = ds.arrays["T"]
    Test.@test length(size(t_arr)) > 0
    
    @info "Completed Google Cloud Zarr metadata test successfully!"
end
