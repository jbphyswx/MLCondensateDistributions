using Test: Test
using DataFrames: DataFrames
using Arrow: Arrow
using MLCondensateDistributions: MLCondensateDistributions as MLCD

# Integration test: remote GoogleLES → Arrow file → DataFrame columns match package schema.
# Uses `MLCD.GoogleLES.build_tabular` and `MLCD.DatasetBuilder.SCHEMA_SYMBOL_ORDER` (package API; no `include` of utils).

Test.@testset "End-to-End GCP Orchestrator Integration Pipeline" begin
    @info "Starting the live integration deployment test targeting GoogleLES Cloud Datasets..."

    # Persistent directory under `test/` so you can inspect artifacts after a run.
    output_dir = joinpath(@__DIR__, "test_output")
    mkpath(output_dir)

    # Drop stale Arrow shards so schema/name checks reflect this run only.
    for f in readdir(output_dir; join=true)
        if endswith(f, ".arrow")
            rm(f)
        end
    end

    # Validation 1: build one timestep from the orchestrator and confirm the expected file appears.
    # `max_timesteps=1` keeps the integration cheap while still exercising the full write path.
    @info "Dispatching `MLCD.GoogleLES.build_tabular` for a single timestep sample directly from remote server..."
    MLCD.GoogleLES.build_tabular(10, 1, "amip", output_dir; max_timesteps=1)

    target_arrow = joinpath(output_dir, "googleles_case__10__month__1__exp__amip.arrow")
    Test.@test isfile(target_arrow)
    @info "Successfully found serialized payload at: $target_arrow"

    # Validation 2: read the table back with Arrow/DataFrames and compare column names to the canonical schema.
    @info "Reconstituting structured representation vector..."
    table = Arrow.Table(target_arrow)
    df_reinstated = DataFrames.DataFrame(table)

    # Dump the first 15 mapped rows into the log so the column layout is visible when debugging CI or local runs.
    @info "Here is exactly how the extracted Tabular Physics array manifests inside the integration payload:"
    display(DataFrames.first(df_reinstated, 15))

    # Validate structure: must match `dataset_spec.md` via package canonical names.
    expected_schema = collect(MLCD.DatasetBuilder.DATASET_SPEC_CODE_NAMES)
    Test.@test names(df_reinstated) == expected_schema
    Test.@test names(df_reinstated) == string.(collect(MLCD.DatasetBuilder.SCHEMA_SYMBOL_ORDER))

    # Row count can be 0 on a sparse cloud mask for some chunks; here we still assert schema parity above.
    @info "Integration validated identically against schema. Sliced $(DataFrames.nrow(df_reinstated)) valid cloudy blocks out of the raw sub-grid!"
end
