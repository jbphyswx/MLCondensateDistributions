using Test: Test
using DataFrames: DataFrames
using Parquet2: Parquet2
using MLCondensateDistributions: DataHandling

Test.@testset "load_moments_dataframe" begin
    mktempdir() do d
        df = DataFrames.DataFrame(
            resolution_h = Float32[100.0f0, 200.0f0],
            resolution_z = Float32[50.0f0, 50.0f0],
            cov_qt_w = Float32[0.1f0, 0.2f0],
            var_w = Float32[1.0f0, 2.0f0],
            tke = Float32[0.5f0, 1.0f0],
            data_source = ["A", "B"],
        )
        Parquet2.writefile(joinpath(d, "case.parquet"), df; compression_codec=:snappy)

        out, targets = DataHandling.load_moments_dataframe(d)
        Test.@test sort(targets) == [:cov_qt_w, :var_w]
        Test.@test DataFrames.nrow(out) == 2
        Test.@test :tke ∉ targets

        out2, targets2 = DataHandling.load_moments_dataframe(d; extra_columns=[:tke, :data_source])
        Test.@test targets2 == targets
        Test.@test hasproperty(out2, :tke)
        Test.@test hasproperty(out2, :data_source)
    end
end
