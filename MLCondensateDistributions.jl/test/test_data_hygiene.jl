using Test: Test
using DataFrames: DataFrames
using Parquet2: Parquet2

include("../utils/coarse_graining.jl")
include("../utils/dynamics.jl")
include(joinpath(@__DIR__, "..", "utils", "statistical_methods", "StatisticalMethods.jl"))
include("../utils/dataset_builder.jl")
include("../utils/dataloader.jl")

using .DatasetBuilder: DatasetBuilder

Test.@testset "Dataset builder drops non-finite cells" begin
    dims = (8, 8, 6)
    fine_fields = Dict{String, AbstractArray{Float32, 3}}()

    for k in ["ta", "hus", "wa", "ua", "va", "thetali", "pfull", "rhoa"]
        fine_fields[k] = rand(Float32, dims...)
    end

    # Seed cloud so rows survive condensate masking.
    fine_fields["clw"] = fill(0.01f0, dims...)
    fine_fields["cli"] = fill(0.005f0, dims...)

    # Inject a non-finite source value that should be excluded from output rows.
    fine_fields["hus"][1, 1, 1] = NaN32

    metadata = Dict{Symbol, Any}(
        :data_source => "Synth",
        :month => 1,
        :cfSite_number => 42,
        :forcing_model => "Test",
        :experiment => "amip",
        :timestep => 0,
    )

    spatial_info = Dict{Symbol, Any}(
        :dx_native => 100.0f0,
        :domain_h => 800.0f0,
        :min_dh => 100.0f0,
        :dz_native_profile => fill(50.0f0, dims[3]),
        :coarsening_mode => :hybrid,
    )

    max_dz = maximum(spatial_info[:dz_native_profile])
    df = DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info, max_dz)

    Test.@test DataFrames.nrow(df) > 0
    real_cols = [nm for nm in names(df) if eltype(df[!, nm]) <: Real]
    has_nonfinite = any(any(x -> !isfinite(x), df[!, nm]) for nm in real_cols)
    Test.@test !has_nonfinite
end

Test.@testset "Loader prunes parquet files with non-finite values" begin
    mktempdir() do d
        valid_file = joinpath(d, "valid.parquet")
        invalid_file = joinpath(d, "invalid.parquet")

        valid_df = DataFrames.DataFrame(qt = Float32[0.1, 0.2], q_liq = Float32[0.01, 0.02])
        invalid_df = DataFrames.DataFrame(qt = Float32[0.1, NaN32], q_liq = Float32[0.01, 0.02])

        Parquet2.writefile(valid_file, valid_df; compression_codec=:snappy)
        Parquet2.writefile(invalid_file, invalid_df; compression_codec=:snappy)

        keep = prune_incompatible_parquet_files!(d, [:qt, :q_liq]; verbose=false)

        Test.@test keep == [valid_file]
        Test.@test isfile(valid_file)
        Test.@test !isfile(invalid_file)
    end
end
