using Test: Test
using DataFrames: DataFrames
using Statistics: Statistics

include("../utils/coarse_graining.jl")
include("../utils/dynamics.jl")
include("../utils/dataset_builder.jl")

using .DatasetBuilder: DatasetBuilder

Test.@testset "Arrow schema matches dataset_spec.md (36 columns)" begin
    Test.@test length(DatasetBuilder.SCHEMA_SYMBOL_ORDER) == 36
    Test.@test length(DatasetBuilder.DATASET_SPEC_CODE_NAMES) == 36
    Test.@test string.(collect(DatasetBuilder.SCHEMA_SYMBOL_ORDER)) ==
        collect(DatasetBuilder.DATASET_SPEC_CODE_NAMES)
end

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
        :domain_h => 6000.0f0,
        :min_h_resolution => 100.0f0,
        :dz_native_profile => fill(25.0f0, 16), # 16 vertical levels to match dims
        :coarsening_mode => :hybrid,
    )
    
    @info "Piping through process_abstract_chunk..."
    df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
    
    # Verification
    Test.@test df isa DataFrames.DataFrame
    
    # Multi-level horizontal coarse graining now emits all levels at or above
    # `:min_h_resolution`. This synthetic setup should produce trainable rows.
    Test.@test DataFrames.nrow(df) > 0
    
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
    Test.@test all(df.domain_h .== 6000.0f0)
    Test.@test all(df.tke .>= -1f-6)
    Test.@test all(abs.(df.q_con .- (df.q_liq .+ df.q_ice)) .<= 1f-10)
    Test.@test all((0f0 .<= df.liq_fraction) .& (df.liq_fraction .<= 1f0))
    Test.@test all((0f0 .<= df.ice_fraction) .& (df.ice_fraction .<= 1f0))
    Test.@test all((0f0 .<= df.cloud_fraction) .& (df.cloud_fraction .<= 1f0))
    Test.@test all(df.cloud_fraction .>= df.liq_fraction)
    Test.@test all(df.cloud_fraction .>= df.ice_fraction)

    # Multiscale invariants: we should emit multiple horizontal and vertical levels.
    h_levels = sort(unique(df.resolution_h))
    z_levels = sort(unique(df.resolution_z))
    Test.@test length(h_levels) > 1
    Test.@test length(z_levels) > 1
    Test.@test Statistics.minimum(h_levels) >= 100.0f0
    Test.@test Statistics.maximum(z_levels) <= 400.0f0

    # Ensure (resolution_h, resolution_z) combinations are present.
    hz_pairs = unique(zip(df.resolution_h, df.resolution_z))
    Test.@test length(hz_pairs) > max(length(h_levels), length(z_levels))
    
    @info "Tabular Extraction and Sparsity Masks successfully validated!"
end

Test.@testset "GoogleLES metadata schema regression" begin
    dims = (16, 16, 16)
    fine_fields = Dict{String, AbstractArray{Float32, 3}}()

    for k in ["ta", "hus", "wa", "ua", "va", "thetali", "pfull", "rhoa"]
        fine_fields[k] = rand(Float32, dims...)
    end

    q_c = zeros(Float32, dims...)
    q_c[1:4, 1:4, 5] .= 0.05f0
    fine_fields["clw"] = q_c
    fine_fields["cli"] = zeros(Float32, dims...)

    metadata = Dict{Symbol, Any}(
        :data_source => "GoogleLES",
        :month => 1,
        :site_id => 10,
        :cfSite_number => 10,
        :forcing_model => "GoogleLES",
        :experiment => "amip",
    )

    spatial_info = Dict{Symbol, Any}(
        :dx_native => 50.0f0,
        :domain_h => 6000.0f0,
        :min_h_resolution => 100.0f0,
        :dz_native_profile => fill(25.0f0, 16),
        :coarsening_mode => :hybrid,
    )

    df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
    Test.@test df isa DataFrames.DataFrame
    Test.@test names(df) == string.(collect(DatasetBuilder.SCHEMA_SYMBOL_ORDER))
    Test.@test all(df.data_source .== "GoogleLES")
    Test.@test all(df.cfSite_number .== 10)
    Test.@test all(df.domain_h .== 6000.0f0)
    Test.@test all(df.tke .>= -1f-6)
    Test.@test all(abs.(df.q_con .- (df.q_liq .+ df.q_ice)) .<= 1f-10)
    Test.@test all((0f0 .<= df.liq_fraction) .& (df.liq_fraction .<= 1f0))
    Test.@test all((0f0 .<= df.ice_fraction) .& (df.ice_fraction .<= 1f0))
    Test.@test all((0f0 .<= df.cloud_fraction) .& (df.cloud_fraction .<= 1f0))

    # Regression guard: vertical resolution ladder should be present.
    z_levels = sort(unique(df.resolution_z))
    Test.@test Statistics.minimum(z_levels) == 25.0f0
    Test.@test Statistics.maximum(z_levels) == 400.0f0
    Test.@test length(z_levels) >= 3
end
