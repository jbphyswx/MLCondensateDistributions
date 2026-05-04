using Lux: Lux
using Optimisers: Optimisers
using Random: Random
using Statistics: Statistics
using Zygote: Zygote
using Serialization: Serialization

"""
Training note (current implementation):

- Features are standardized before training.
- Targets are standardized before training.
- For phase targets (`q_liq`, `q_ice`), training uses a two-head output per target:
    1) magnitude head (regression in standardized target space),
    2) presence head (binary logit for nonzero vs zero).
- The phase loss is a weighted sum of:
    - BCE-with-logits on presence for all samples,
    - MSE on magnitude only for samples where the phase is present.
- Non-phase targets (`var_*`, `cov_*`) use MSE in standardized space.
- Early stopping tracks test loss and restores the best validation checkpoint
    before saving model artifacts.

At decode time, phase predictions are mapped back to physical space by:
- de-standardizing magnitude,
- applying presence probability threshold,
- forcing exact zero when presence is predicted off,
- clamping phase magnitudes to nonnegative values.
"""

function _target_metrics(y_true::AbstractVector{<:Real}, y_pred::AbstractVector{<:Real})
    err = y_pred .- y_true
    mse = Statistics.mean(abs2, err)
    rmse = sqrt(mse)
    mae = Statistics.mean(abs, err)
    denom = sum(abs2, y_true .- Statistics.mean(y_true))
    r2 = denom == 0 ? NaN : (1 - sum(abs2, err) / denom)
    return (mse=mse, rmse=rmse, mae=mae, r2=r2)
end

function _finite_sample_mask(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})
    n = size(X, 2)
    keep = trues(n)
    @inbounds for i in 1:n
        keep[i] = all(isfinite, @view X[:, i]) && all(isfinite, @view Y[:, i])
    end
    return keep
end

function _standardize_targets(Y::AbstractMatrix{<:Real})
    μy = Float32.(Statistics.mean(Y; dims=2))
    σy = Float32.(Statistics.std(Y; dims=2, corrected=false))
    @inbounds for i in eachindex(σy)
        if !(isfinite(σy[i]) && σy[i] > 0f0)
            σy[i] = 1f0
        end
    end
    Y_std = (Y .- μy) ./ σy
    return Y_std, μy, σy
end

@inline function _destandardize_targets(Y_std::AbstractMatrix{<:Real}, μy::AbstractMatrix{<:Real}, σy::AbstractMatrix{<:Real})
    return Y_std .* σy .+ μy
end

@inline function _is_phase_target(target::Symbol)
    return target in (:q_liq, :q_ice)
end

function _build_output_mapping(target_cols::Vector{Symbol})
    n_targets = length(target_cols)
    magnitude_idx = zeros(Int, n_targets)
    presence_idx = zeros(Int, n_targets)
    next_out = 1
    for i in 1:n_targets
        magnitude_idx[i] = next_out
        if _is_phase_target(target_cols[i])
            presence_idx[i] = next_out + 1
            next_out += 2
        else
            next_out += 1
        end
    end
    return magnitude_idx, presence_idx, next_out - 1
end

@inline function _sigmoid(x::Float32)
    if x >= 0f0
        z = exp(-x)
        return 1f0 / (1f0 + z)
    else
        z = exp(x)
        return z / (1f0 + z)
    end
end

@inline function _bce_with_logits(logit::Float32, target01::Float32)
    return max(logit, 0f0) - logit * target01 + log1p(exp(-abs(logit)))
end

function _presence_magnitude_loss(
    y_pred_out::AbstractMatrix{<:Real},
    y_true_std::AbstractMatrix{<:Real},
    y_true_raw::AbstractMatrix{<:Real},
    target_cols::Vector{Symbol},
    magnitude_idx::Vector{Int},
    presence_idx::Vector{Int};
    presence_threshold::Float32 = 0f0,
    presence_loss_weight::Float32 = 1f0,
    magnitude_loss_weight::Float32 = 1f0,
)
    T = Float32
    total = zero(T)
    weight_sum = zero(T)

    for i in eachindex(target_cols)
        pred_mag = @view y_pred_out[magnitude_idx[i], :]
        true_std = @view y_true_std[i, :]
        target = target_cols[i]

        if _is_phase_target(target)
            pred_logit = @view y_pred_out[presence_idx[i], :]
            true_raw = @view y_true_raw[i, :]

            n = length(true_raw)
            y01 = ifelse.(true_raw .> presence_threshold, one(T), zero(T))

            # BCE-with-logits over all samples for phase presence.
            bce = Statistics.mean(max.(pred_logit, zero(T)) .- pred_logit .* y01 .+ log1p.(exp.(-abs.(pred_logit))))
            total += presence_loss_weight * T(bce)
            weight_sum += presence_loss_weight

            # Magnitude loss only over present samples; avoid masking allocations.
            n_present = sum(y01)
            if n_present > 0
                d = pred_mag .- true_std
                mse_present = sum((d .* d) .* y01) / n_present
                total += magnitude_loss_weight * T(mse_present)
                weight_sum += magnitude_loss_weight
            end
        else
            total += T(Statistics.mean(abs2, pred_mag .- true_std))
            weight_sum += 1f0
        end
    end

    return total / max(weight_sum, eps(Float32))
end

function _decode_predictions(
    y_pred_out::AbstractMatrix{<:Real},
    μy::AbstractMatrix{<:Real},
    σy::AbstractMatrix{<:Real},
    target_cols::Vector{Symbol},
    magnitude_idx::Vector{Int},
    presence_idx::Vector{Int};
    presence_prob_threshold::Float32 = 0.5f0,
)
    n_targets = length(target_cols)
    n_samples = size(y_pred_out, 2)
    Y = Matrix{Float32}(undef, n_targets, n_samples)

    @inbounds for i in 1:n_targets
        μ = Float32(μy[i])
        σ = Float32(σy[i])
        target = target_cols[i]
        for j in 1:n_samples
            mag_std = Float32(y_pred_out[magnitude_idx[i], j])
            mag_phys = mag_std * σ + μ
            if _is_phase_target(target)
                p = _sigmoid(Float32(y_pred_out[presence_idx[i], j]))
                Y[i, j] = p >= presence_prob_threshold ? max(0f0, mag_phys) : 0f0
            else
                Y[i, j] = mag_phys
            end
        end
    end

    return Y
end

@inline function _clamp_physical_constraints(Y_pred::AbstractMatrix{<:Real}, target_cols::Vector{Symbol})
    Y_clipped = copy(Y_pred)
    @inbounds for (i, target) in enumerate(target_cols)
        if target in (:q_liq, :q_ice)  # condensate quantities must be >= 0
            for j in 1:size(Y_clipped, 2)
                if Y_clipped[i, j] < 0f0
                    Y_clipped[i, j] = 0f0
                end
            end
        end
    end
    return Y_clipped
end

"""
        train_model(data_dir::String; kwargs...)

Train the Lux model on processed Arrow data.

Important behavior:
- `q_liq` and `q_ice` use presence+magnitude modeling via
    `phase_presence_threshold`, `phase_presence_loss_weight`,
    `phase_magnitude_loss_weight`, and `phase_presence_prob_threshold`.
- Early stopping uses `early_stopping_patience` and `early_stopping_min_delta`.
- The best-validation checkpoint is restored before model/artifact save.

Returns a named tuple with model state, scaling statistics, metrics,
and diagnostics path.
"""
function train_model(
    data_dir::String;
    feature_cols::Vector{Symbol} = [:qt, :theta_li, :p, :rho, :w, :tke, :domain_h, :resolution_z],
    target_cols::Vector{Symbol} = [:q_liq, :q_ice],
    epochs::Int = 100,
    lr::Float64 = 1e-3,
    batch_size::Int = 512,
    train_fraction::Float64 = 0.8,
    seed::Int = 123,
    hidden_layers::Vector{Int} = [64, 64, 32],
    model_name::String = "lux_model.jls",
    run_artifact_name::String = "lux_run_artifact.jls",
    save_run_artifact::Bool = true,
    write_diagnostics::Bool = true,
    early_stopping_patience::Int = 15,
    early_stopping_min_delta::Float64 = 0.0,
    phase_presence_threshold::Float32 = 0f0,
    phase_presence_loss_weight::Float32 = 1f0,
    phase_magnitude_loss_weight::Float32 = 1f0,
    phase_presence_prob_threshold::Float32 = 0.5f0,
    log_every_n_batches::Int = 100,
)
    train_fraction <= 0 || train_fraction >= 1 && error("train_fraction must be strictly between 0 and 1")

    @info "Loading data from $data_dir..."
    X_raw, Y_raw = load_processed_data(data_dir, feature_cols, target_cols)

    keep_mask = _finite_sample_mask(X_raw, Y_raw)
    dropped = count(!, keep_mask)
    if dropped > 0
        @warn "Dropping non-finite samples before training" dropped total=size(X_raw, 2)
        X_raw = X_raw[:, keep_mask]
        Y_raw = Y_raw[:, keep_mask]
    end

    size(X_raw, 2) > 1 || error("Not enough finite samples after filtering to run train/test split.")

    @info "Standardizing features..."
    X, μ, σ = standardize_data(X_raw)
    Y, μy, σy = _standardize_targets(Y_raw)

    n_samples = size(X, 2)
    n_train = clamp(floor(Int, train_fraction * n_samples), 1, n_samples - 1)

    rng = Random.default_rng()
    Random.seed!(rng, seed)
    shuffled = Random.shuffle(rng, collect(1:n_samples))
    train_idx = shuffled[1:n_train]
    test_idx = shuffled[(n_train + 1):end]

    X_train = X[:, train_idx]
    Y_train_std = Y[:, train_idx]
    Y_train_raw = Y_raw[:, train_idx]
    X_test = X[:, test_idx]
    Y_test_std = Y[:, test_idx]
    Y_test = Y_raw[:, test_idx]
    X_test_raw = X_raw[:, test_idx]

    @info "Training samples=$(length(train_idx)); test samples=$(length(test_idx))"

    magnitude_idx, presence_idx, model_output_dim = _build_output_mapping(target_cols)
    model = CondensateMLP(length(feature_cols), model_output_dim, hidden_layers)
    ps, st = Lux.setup(rng, model)

    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)

    @info "Starting training on $(length(train_idx)) samples..."
    train_loss_history = Float64[]
    test_loss_history = Float64[]

    best_test_loss = Inf
    best_epoch = 0
    best_ps = deepcopy(ps)
    patience_counter = 0

    n_train_samples = size(X_train, 2)
    for epoch in 1:epochs
        epoch_started = time()
        perm = Random.shuffle(rng, 1:n_train_samples)
        epoch_loss = 0.0
        n_batches = 0
        expected_batches = cld(n_train_samples, batch_size)

        for i in 1:batch_size:n_train_samples
            idx = perm[i:min(i + batch_size - 1, n_train_samples)]
            x_batch = X_train[:, idx]
            y_batch_std = Y_train_std[:, idx]
            y_batch_raw = Y_train_raw[:, idx]

            loss_fn = (p) -> begin
                y_pred_out, _ = model(x_batch, p, st)
                return _presence_magnitude_loss(
                    y_pred_out,
                    y_batch_std,
                    y_batch_raw,
                    target_cols,
                    magnitude_idx,
                    presence_idx;
                    presence_threshold=phase_presence_threshold,
                    presence_loss_weight=phase_presence_loss_weight,
                    magnitude_loss_weight=phase_magnitude_loss_weight,
                )
            end

            loss, grads = Zygote.withgradient(loss_fn, ps)
            if !isfinite(loss)
                @warn "Encountered non-finite batch loss; skipping batch" epoch batch_start=i
                continue
            end
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

            epoch_loss += loss
            n_batches += 1

            # if (n_batches % log_every_n_batches == 0) || (n_batches == expected_batches)
                # elapsed = time() - epoch_started
                # @info "Epoch $(epoch) batch $(n_batches)/$(expected_batches)" batch_loss=loss elapsed_s=round(elapsed; digits=1)
            # end
        end

        n_batches > 0 || error("All batches in epoch $(epoch) had non-finite loss; aborting training.")

        epoch_mean_loss = epoch_loss / n_batches
        push!(train_loss_history, epoch_mean_loss)

        y_test_pred_out, _ = model(X_test, ps, st)
        test_loss = _presence_magnitude_loss(
            y_test_pred_out,
            Y_test_std,
            Y_test,
            target_cols,
            magnitude_idx,
            presence_idx;
            presence_threshold=phase_presence_threshold,
            presence_loss_weight=phase_presence_loss_weight,
            magnitude_loss_weight=phase_magnitude_loss_weight,
        )
        push!(test_loss_history, test_loss)

        if epoch % 10 == 0 || epoch == 1
            @info "Epoch $epoch: train_loss=$(epoch_mean_loss), test_loss=$(test_loss) | best_test_loss=$(best_test_loss), best_epoch=$(best_epoch), patience_counter=$(patience_counter)/$(early_stopping_patience)"
        end

        if test_loss < (best_test_loss - early_stopping_min_delta)
            best_test_loss = test_loss
            best_epoch = epoch
            best_ps = deepcopy(ps)
            patience_counter = 0
        else
            patience_counter += 1
            if patience_counter >= early_stopping_patience
                @info "Early stopping triggered after epoch $epoch (best_test_loss=$(best_test_loss), best_epoch=$(best_epoch), min_delta=$(early_stopping_min_delta))"
                break
            end
        end
    end

    ps = best_ps
    @info "Restored best validation checkpoint" best_epoch best_test_loss

    model_dir = Paths.model_data_dir()
    mkpath(model_dir)

    save_path = joinpath(model_dir, model_name)
    Serialization.serialize(save_path, (
        model=model,
        ps=ps,
        st=st,
        μ=μ,
        σ=σ,
        μy=μy,
        σy=σy,
        feature_cols=feature_cols,
        target_cols=target_cols,
        model_output_dim=model_output_dim,
        magnitude_idx=magnitude_idx,
        presence_idx=presence_idx,
        train_fraction=train_fraction,
        seed=seed,
        best_epoch=best_epoch,
        best_test_loss=best_test_loss,
    ))
    @info "Model saved to $save_path"

    y_test_pred_out, _ = model(X_test, ps, st)
    y_test_pred = _decode_predictions(
        y_test_pred_out,
        μy,
        σy,
        target_cols,
        magnitude_idx,
        presence_idx;
        presence_prob_threshold=phase_presence_prob_threshold,
    )
    metrics = Dict{Symbol, NamedTuple}()
    for (i, target) in enumerate(target_cols)
        metrics[target] = _target_metrics(vec(Y_test[i, :]), vec(y_test_pred[i, :]))
    end

    for target in target_cols
        m = metrics[target]
        @info "Target $(target): rmse=$(m.rmse), mae=$(m.mae), r2=$(m.r2)"
    end

    if save_run_artifact
        artifact_path = joinpath(model_dir, run_artifact_name)
        Serialization.serialize(artifact_path, (
            feature_cols=feature_cols,
            target_cols=target_cols,
            train_idx=train_idx,
            test_idx=test_idx,
            X_test_raw=X_test_raw,
            X_test_std=X_test,
            Y_test=Y_test,
            Y_test_std=Y_test_std,
            Y_test_pred=y_test_pred,
            Y_test_pred_out=y_test_pred_out,
            train_loss_history=train_loss_history,
            test_loss_history=test_loss_history,
            metrics=metrics,
            best_epoch=best_epoch,
            best_test_loss=best_test_loss,
            magnitude_idx=magnitude_idx,
            presence_idx=presence_idx,
            model_path=save_path,
        ))
        @info "Run artifact saved to $artifact_path"
    end

    diagnostics_dir = nothing
    if write_diagnostics
        diagnostics_dir = write_training_diagnostics(model, ps, st, X_test_raw, X_test, Y_test, feature_cols, target_cols, test_loss_history, model_dir)
        if isnothing(diagnostics_dir)
            @info "Diagnostics extension not loaded; skipping plots. Load CairoMakie to enable plotting diagnostics."
        else
            @info "Diagnostics saved to $diagnostics_dir"
        end
    end

    return (model=model, ps=ps, st=st, μ=μ, σ=σ, μy=μy, σy=σy, best_epoch=best_epoch, best_test_loss=best_test_loss, metrics=metrics, diagnostics_dir=diagnostics_dir)
end

# Check if script is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    train_model(Paths.experiment_data_dir("amip_baseline"))
end
