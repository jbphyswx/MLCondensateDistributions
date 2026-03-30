using Pkg; Pkg.activate(".")
include("../../utils/build_training_data.jl")

# Research Workflow: Generate a 5-timestep sample for the AMIP experiment (Site 0, Month 1)
# This serves as the benchmark data for the baseline MLP model.

output_dir = "data/amip_baseline"
mkpath(output_dir)

println("--- Starting AMIP Baseline Data Generation ---")
println("Site: 0, Month: 1, Experiment: amip")
println("Output Directory: $output_dir")

build_training_data(
    0,                  # cfSite_number
    1,                  # month
    "amip",             # experiment_name
    output_dir;         # output_dir
    max_timesteps=5     # limit for initial baseline research
)

println("--- Data Generation Complete ---")
