"""
Full AMIP training experiment.

Usage (REPL-friendly):
    include("full_train_experiment.jl")
    run_full_training!()
    run_full_training!(make_plots=true)

Training and plotting are intentionally separated.
Use `plot_full_training_diagnostics.jl` after training.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)

using MLCondensateDistributions: MLCondensateDistributions as MLCD

const FULL_FEATURE_COLS = [:qt, :theta_li, :p, :rho, :w, :tke, :domain_h, :resolution_z]
const FULL_TARGET_COLS = [
    :q_liq,
    :q_ice,
    :var_qt,
    :var_ql,
    :var_qi,
    :var_w,
    :var_h,
    :cov_qt_ql,
    :cov_qt_qi,
    :cov_qt_w,
    :cov_qt_h,
    :cov_ql_qi,
    :cov_ql_w,
    :cov_ql_h,
    :cov_qi_w,
    :cov_qi_h,
    :cov_w_h,
]

function run_full_training!(;
    epochs::Int = Int(1e6), # very high number, just stop when validation loss plateaus with early stopping
    lr::Float64 = 1e-3,
    batch_size::Int = 2048,
    train_fraction::Float64 = 0.9,
    seed::Int = 123,
    make_plots::Bool = false,
    early_stopping_patience::Int = 100,
)
    data_dir = MLCD.Paths.processed_data_root()
    @info "Running full training experiment" data_dir epochs lr batch_size train_fraction seed make_plots early_stopping_patience

    result = MLCD.train_model(
        data_dir;
        feature_cols=FULL_FEATURE_COLS,
        target_cols=FULL_TARGET_COLS,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        train_fraction=train_fraction,
        seed=seed,
        model_name="lux_full_model.jls",
        run_artifact_name="lux_full_run_artifact.jls",
        save_run_artifact=true,
        write_diagnostics=false,
        early_stopping_patience=early_stopping_patience,
    )

    diagnostics_dir = nothing
    if make_plots
        include("plot_full_training_diagnostics.jl")
        diagnostics_dir = plot_full_training_diagnostics!()
        @info "Full training diagnostics generated" diagnostics_dir
    end

    return (; result..., diagnostics_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_training!()
end