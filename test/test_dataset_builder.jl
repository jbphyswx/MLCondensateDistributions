using Test: Test
using DataFrames: DataFrames

include("../utils/GoogleLES.jl")
include("../utils/coarse_graining.jl")
include("../utils/dynamics.jl")
include("../utils/dataset_builder.jl")

using .DatasetBuilder: DatasetBuilder

Test.@testset "Dataset Builder Orchestration Unit Test" begin
    @info "Synthesizing a dummy 16x16x16 chunk of fine-scale LES fields..."
    
    # 16x16x16 block
    dims = (16, 16, 16)
    fine_fields = Dict{String, AbstractArray{Float32, 3}}()
    
    # Fill fields with random numbers
    for k in ["ta", "hus", "wa", "ua", "va", "thetali", "pfull", "rhoa"]
        fine_fields[k] = rand(Float32, dims...)
    end
    
    # Create totally empty cloud grids (all zeros)
    cloud_w = zeros(Float32, dims...)
    cloud_i = zeros(Float32, dims...)
    
    # Inject a thick cloud into only a 4x4 horizontal region on depth 5
    cloud_w[1:4, 1:4, 5] .= 0.05f0
    cloud_i[1:4, 1:4, 5] .= 0.01f0
    
    fine_fields["clw"] = cloud_w
    fine_fields["cli"] = cloud_i
    
    metadata = Dict{Symbol, Any}(
        :data_source => "Synth",
        :month => 999,
        :cfSite_number => 1,
        :forcing_model => "Test",
        :experiment => "amip"
    )
    
    spatial_info = Dict{Symbol, Any}(
        :dx_native => 50.0f0,
        :dz_native_profile => fill(25.0f0, 16) # 16 vertical levels to match dims
    )
    
    @info "Piping through process_abstract_chunk..."
    df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
    
    # Verification
    Test.@test df isa DataFrames.DataFrame
    
    # The block 16x16x16 was coarse grained horizontally by 2x2.
    # CG output dimensions should be 8x8x16. Total cells = 1024.
    # We only injected condensate in a 4x4 area in the fine grid (which corresponds to 
    # a 2x2 area in the coarse grid) down at index 5.
    # A 2x2 patch = 4 cells. Those 4 cells have condensate. The other 1020 cells DO NOT.
    # The sparsity mask drops cells where mean(q) < 1e-8.
    # Hence, the resulting DataFrame should ONLY have EXACTLY 4 rows!
    
    Test.@test DataFrames.nrow(df) == 4
    
    expected_schema = string.(collect(DatasetBuilder.SCHEMA_SYMBOL_ORDER))
    
    # Strictly enforce that absolutely NO extraneous columns exist and absolutely NO variables were dropped natively
    actual_columns = names(df)
    missing_cols = setdiff(expected_schema, actual_columns)
    extraneous_cols = setdiff(actual_columns, expected_schema)
    
    Test.@test isempty(missing_cols)
    Test.@test isempty(extraneous_cols)
    Test.@test actual_columns == expected_schema
    
    # Ensure they have mapping matching the test_id metadata bounds natively
    Test.@test df.month[1] == 999
    
    @info "Tabular Extraction and Sparsity Masks successfully validated!"
end
