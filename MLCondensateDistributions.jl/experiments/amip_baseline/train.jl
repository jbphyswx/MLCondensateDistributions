using Pkg: Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
include("../../utils/train_lux.jl")

data_dir = MLCondensateDistributions.Paths.processed_data_root()
epochs = 100
lr = 1e-3
batch_size = 512

println("--- [Experiment: AMIP Baseline] Training Lux Model ---")
println("Data source: $(data_dir)")
println("Hyperparameters: epochs=$(epochs), lr=$(lr), batch_size=$(batch_size)")

train_model(data_dir; epochs=epochs, lr=lr, batch_size=batch_size)

println("--- Training Complete ---")
