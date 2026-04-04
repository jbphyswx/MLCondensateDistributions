using Test: Test
using Random: Random

include("../utils/coarse_graining.jl")
include("../utils/dynamics.jl")
include("../utils/dataset_builder.jl")
include("../utils/dataset_builder_impl.jl")

Test.@testset "DatasetBuilderImpl parity and allocation guard" begin
    Random.seed!(42)

    nx, ny, nz = 124, 124, 120
    fine_fields = Dict{String, Array{Float32, 3}}(
        "hus" => rand(Float32, nx, ny, nz) .* 1f-2,
        "thetali" => rand(Float32, nx, ny, nz) .* 300f0,
        "ta" => rand(Float32, nx, ny, nz) .* 300f0,
        "pfull" => rand(Float32, nx, ny, nz) .* 1f5,
        "rhoa" => rand(Float32, nx, ny, nz),
        "ua" => randn(Float32, nx, ny, nz),
        "va" => randn(Float32, nx, ny, nz),
        "wa" => randn(Float32, nx, ny, nz),
        "clw" => rand(Float32, nx, ny, nz) .* 1f-5,
        "cli" => rand(Float32, nx, ny, nz) .* 1f-5,
    )

    metadata = Dict{Symbol, Any}(
        :data_source => "bench",
        :month => 1,
        :cfSite_number => 1,
        :forcing_model => "bench",
        :experiment => "bench",
        :verbose => false,
    )

    spatial_info = Dict{Symbol, Any}(
        :dx_native => Float32(50),
        :domain_h => Float32((nx - 1) * 50),
        :min_h_resolution => Float32(1000),
        :dz_native_profile => fill(Float32(20), nz),
    )

    # Warmup
    df_old = DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
    df_new = DatasetBuilderImpl.process_abstract_chunk_impl(fine_fields, metadata, spatial_info)

    Test.@test size(df_old, 1) == size(df_new, 1)

    alloc_old = @allocated DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
    alloc_new = @allocated DatasetBuilderImpl.process_abstract_chunk_impl(fine_fields, metadata, spatial_info)

    # Hard guard: prevent a return to the multi-GB regression.
    Test.@test alloc_new < 100_000_000

    # Soft relative guard: v2 should remain in the same order of magnitude as old.
    Test.@test alloc_new < 5 * alloc_old
end
