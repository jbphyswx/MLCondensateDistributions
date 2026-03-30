using Pkg; Pkg.activate(".")
include("../../utils/train_lux.jl")

# Research Workflow: Train the baseline Lux model on the AMIP baseline data.
# This script records the hyperparameters used for the initial model version.

data_dir = "data/amip_baseline"
epochs = 100
lr = 1e-3
batch_size = 512

println("--- Starting AMIP Baseline Training ---")
println("Data Directory: $data_dir")
println("Epochs: $epochs, Batch Size: $batch_size, Learning Rate: $lr")

train_model(
    data_dir; 
    epochs=epochs, 
    lr=lr,
    batch_size=batch_size
)

println("--- Training Complete ---")
