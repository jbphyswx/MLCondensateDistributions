using Pkg; Pkg.activate(".")
using MLCondensateDistributions

# Experiment Workflow: Baseline Lux Training
# This script records the training hyperparameters for the AMIP baseline model.

data_dir = "data/amip_baseline"
epochs = 100
lr = 1e-3
batch_size = 512

println("--- [Experiment: AMIP Baseline] Training Lux Model ---")
println("Data source: $data_dir")
println("Hyperparameters: epochs=$epochs, lr=$lr, batch_size=$batch_size")

train_model(
    data_dir; 
    epochs=epochs, 
    lr=lr,
    batch_size=batch_size
)

println("--- Training Complete ---")
