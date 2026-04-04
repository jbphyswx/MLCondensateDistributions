"""
AMIP baseline bootstrap training workflow.

This script incrementally builds a non-empty GoogleLES dataset across
candidate site/month pairs, then trains a Lux model.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)

using Arrow: Arrow
using DataFrames: DataFrames
using Dates: Dates
using CairoMakie: CairoMakie
using MLCondensateDistributions: MLCondensateDistributions as MLCD

include("generate_data.jl")

const OUTPUT_DIR = MLCD.Paths.processed_data_root()
const CACHE_PATH = MLCD.WorkflowState.checked_cases_path("amip_baseline")
const DEFAULT_CANDIDATE_SITES = collect(0:499)
const DEFAULT_CANDIDATE_MONTHS = (1, 4, 7, 10)
const EXPERIMENT = "amip"
const TRAIN_FEATURE_COLS = (:qt, :theta_li, :p, :rho, :w, :tke, :domain_h, :resolution_z)
const TRAIN_TARGET_COLS = (:q_liq, :q_ice)

"""
    total_rows(data_dir)

Count rows across Arrow files in `data_dir`.
Returns 0 when no Arrow files exist.
"""
function total_rows(data_dir::String)
    if !isdir(data_dir)
        return 0
    end
    files = MLCD.prune_incompatible_arrow_files!(data_dir, vcat(TRAIN_FEATURE_COLS, TRAIN_TARGET_COLS); verbose=true)
    if isempty(files)
        return 0
    end
    return sum((DataFrames.nrow(DataFrames.DataFrame(Arrow.Table(f))) for f in files); init=0)
end

function loaded_checked_cases()
    return MLCD.WorkflowState.load_checked_cases(CACHE_PATH)
end

"""
    bootstrap_train!(; kwargs...)

    Collect data until `min_rows` is met, then run model training.

Artifacts:
- Dataset Arrow files: `data/processed`
- Model checkpoint: `data/models/lux_model.jls`
"""
function bootstrap_train!(;
    min_rows::Int = 0,
    max_timesteps_per_case::Int = 0,
    epochs::Int = 20,
    lr::Float64 = 1e-3,
    batch_size::Int = 256,
    candidate_sites::Vector{Int} = MLCD.EnvHelpers.parse_int_list_env("CANDIDATE_SITES", DEFAULT_CANDIDATE_SITES),
    candidate_months::Vector{Int} = MLCD.EnvHelpers.parse_int_list_env("CANDIDATE_MONTHS", DEFAULT_CANDIDATE_MONTHS),
)
    mkpath(OUTPUT_DIR)
    println("Bootstrap output directory: $(OUTPUT_DIR)")

    nrows = total_rows(OUTPUT_DIR)
    println("Existing rows in dataset: $(nrows)")
    checked_cases = loaded_checked_cases()
    target_rows = max(min_rows, 1)

    @info "Scanning $(length(candidate_sites)) sites x $(length(candidate_months)) months for training data"

    for month in candidate_months
        for site in candidate_sites
            if nrows >= target_rows
                break
            end

            key = MLCD.WorkflowState.case_key(site, month, EXPERIMENT)
            if haskey(checked_cases, key)
                cached = checked_cases[key]
                @info "Skipping previously checked case $(key) ($(cached.status)); rows=$(cached.rows), checked_at=$(cached.timestamp)."
                continue
            end

            println("Collecting data for site=$(site), month=$(month)...")
            rows_before = total_rows(OUTPUT_DIR)
            generate_data!(;
                site_id=site,
                month=month,
                experiment=EXPERIMENT,
                output_dir=OUTPUT_DIR,
                max_timesteps=max_timesteps_per_case,
                timestep_batch_size=0,
                include_cfsites=false,
            )

            nrows = total_rows(OUTPUT_DIR)
            rows_added = nrows - rows_before

            if rows_added <= 0
                MLCD.WorkflowState.record_checked_case!(CACHE_PATH, site, month, EXPERIMENT; status="cloudfree", rows=0, timesteps=max_timesteps_per_case)
                checked_cases[key] = (status = "cloudfree", rows = 0, timesteps = max_timesteps_per_case, timestamp = string(Dates.now()))
            else
                MLCD.WorkflowState.record_checked_case!(CACHE_PATH, site, month, EXPERIMENT; status="generated", rows=rows_added, timesteps=max_timesteps_per_case)
                checked_cases[key] = (status = "generated", rows = rows_added, timesteps = max_timesteps_per_case, timestamp = string(Dates.now()))
            end
            println("Rows after site=$(site), month=$(month): $(nrows)")
        end
        if nrows >= target_rows
            break
        end
    end

    if nrows == 0
        error("No trainable rows were produced after scanning all candidate cases. Increase max_timesteps_per_case, lower MIN_H_RESOLUTION, or expand candidate sites/months.")
    end

    println("Training with $(nrows) rows...")
    MLCD.train_model(OUTPUT_DIR; epochs=epochs, lr=lr, batch_size=batch_size)

    model_path = joinpath(MLCD.Paths.model_data_root(), "lux_model.jls")
    println("Model artifact: $(model_path)")
end

bootstrap_train!()
