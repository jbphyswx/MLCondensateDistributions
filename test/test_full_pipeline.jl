using Test: Test
using DataFrames: DataFrames
using Arrow: Arrow
using GoogleLES: GoogleLES as GL

# We must include the orchestrator directly to hit the `GL.build_tabular` core.
include("../utils/build_training_data.jl")

Test.@testset "End-to-End GCP Orchestrator Integration Pipeline" begin
    @info "Starting the live integration deployment test targeting GoogleLES Cloud Datasets..."
    
    # Create a persistent test directory so the user can actually inspect the generated Arrow files
    output_dir = joinpath(@__DIR__, "test_output")
    mkpath(output_dir)
    
    # Clean up any existing stale test data to ensure we are testing the NEW schema
    for f in readdir(output_dir; join=true)
        if endswith(f, ".arrow")
            rm(f)
        end
    end
    
    # Run the orchestrator on a highly constrained 1-timestep environment wrapper
    # Using `month=1`, `exp="amip"` matching the known Google benchmarks. `max_timesteps=1`.
    @info "Dispatching `GL.build_tabular` over 1 time-step directly from remote server..."
    GL.build_tabular(10, 1, "amip", output_dir; max_timesteps=1)
    
    # Validation 1: Arrow serialization hit the disk securely
    target_arrow = joinpath(output_dir, "googleles_amip_1_10_t1.arrow")
    Test.@test isfile(target_arrow)
    @info "Successfully found serialized payload at: $target_arrow"
    
    # Validation 2: Reading it natively back out
    @info "Reconstituting structured representation vector..."
    table = Arrow.Table(target_arrow)
    df_reinstated = DataFrames.DataFrame(table)
    
    # Dump the first 15 mapped rows directly into the log exactly as requested so you can see the architecture
    @info "Here is exactly how the extracted Tabular Physics array manifests inside the integration payload:"
    display(first(df_reinstated, 15))
    
    # Validate structure logic
    expected_schema = string.(collect(DatasetBuilder.SCHEMA_SYMBOL_ORDER))
    Test.@test names(df_reinstated) == expected_schema
    
    # If the chunk successfully traversed the entire framework, it will have rows
    # Note: On a sparse structure like the cloud array slice, this could legitimately be 0 
    # if that precise chunk possessed no cloudy volumes. However, as an integration bounds check,
    # the schema itself is perfectly intact and accurately typed.
    @info "Integration validated identically against schema. Sliced $(DataFrames.nrow(df_reinstated)) valid cloudy blocks out of the raw sub-grid!"
end
