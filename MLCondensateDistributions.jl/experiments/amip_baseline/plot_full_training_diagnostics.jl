"""
Plot diagnostics from a saved training artifact (no retraining).

Usage (REPL-friendly):
    include("plot_full_training_diagnostics.jl")
    plot_full_training_diagnostics!()
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)

using Serialization: Serialization
using MLCondensateDistributions: MLCondensateDistributions as MLCD

function plot_full_training_diagnostics!(;
    artifact_name::String = "lux_full_run_artifact.jls",
    timestamped::Bool = false,
)
    model_dir = MLCD.Paths.model_data_dir()
    artifact_path = joinpath(model_dir, artifact_name)
    isfile(artifact_path) || error("Training artifact not found: $(artifact_path). Run full_train_experiment.jl first.")

    artifact = Serialization.deserialize(artifact_path)
    out_dir = MLCD.Viz.diagnostics_output_dir(model_dir; timestamped=timestamped)
    MLCD.Viz.plot_training_diagnostics_from_artifact(artifact, out_dir)

    println("Full diagnostics saved to $(out_dir)")
    return out_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    plot_full_training_diagnostics!()
end