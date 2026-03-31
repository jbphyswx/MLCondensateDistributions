using Lux: Lux
using Optimisers: Optimisers
using Random: Random
using Statistics: mean
using Zygote: Zygote
using Serialization: serialize

include("dataloader.jl")
include("../src/MLCondensateDistributions.jl")
using .MLCondensateDistributions

function train_model(data_dir::String; epochs=100, lr=1e-3, batch_size=512)
    # 1. Selection of features and targets
    feature_cols = [:qt, :theta_li, :p, :rho, :w, :tke, :resolution_h, :resolution_z]
    target_cols = [:q_liq, :q_ice]
    
    @info "Loading data from $data_dir..."
    X_raw, Y_raw = load_processed_data(data_dir, feature_cols, target_cols)
    
    @info "Standardizing features..."
    X, μ, σ = standardize_data(X_raw)
    Y = Y_raw # Keep targets raw if they are already small, or scale if needed
    
    # 2. Model Initialization
    rng = Random.default_rng()
    Random.seed!(rng, 123)
    
    model = CondensateMLP(length(feature_cols), length(target_cols), [64, 64, 32])
    ps, st = Lux.setup(rng, model)
    
    # 3. Optimizer Setup
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)
    
    # 4. Training Loop
    n_samples = size(X, 2)
    @info "Starting training on $n_samples samples..."
    
    for epoch in 1:epochs
        # Simple shuffling and batching
        perm = Random.shuffle(1:n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in 1:batch_size:n_samples
            idx = perm[i:min(i+batch_size-1, n_samples)]
            x_batch = X[:, idx]
            y_batch = Y[:, idx]
            
            # Loss function: Mean Squared Error
            loss_fn = (p) -> begin
                y_pred, _ = model(x_batch, p, st)
                return mean(abs2, y_pred .- y_batch)
            end
            
            loss, grads = Zygote.withgradient(loss_fn, ps)
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
            
            epoch_loss += loss
            n_batches += 1
        end
        
        if epoch % 10 == 0 || epoch == 1
            @info "Epoch $epoch: Loss = $(epoch_loss / n_batches)"
        end
    end
    
    # 5. Save model and normalization constants
    model_dir = joinpath(dirname(data_dir), "models")
    mkpath(model_dir)
    
    save_path = joinpath(model_dir, "lux_model.jls")
    serialize(save_path, (model=model, ps=ps, st=st, μ=μ, σ=σ))
    @info "Model saved to $save_path"
    
    return model, ps, st, μ, σ
end

# Check if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    train_model("data")
end
