"""
AMIP baseline batch data generation entrypoint.

This script processes site/month cases in batches, optionally in parallel,
and pauses between batches when running interactively so you can decide whether
to continue.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)

using Arrow: Arrow
using DataFrames: DataFrames
using Dates: Dates
using Distributed: Distributed
using Random: Random
using MLCondensateDistributions: MLCondensateDistributions as MLCD

include("generate_data.jl")

# none of these are documented, and these seem like they should be kwarg defaults? not consts
# this feels like abuse of consts for some of the kwargs.
const DEFAULT_CANDIDATE_SITES = collect(0:499)
const DEFAULT_CANDIDATE_MONTHS = [1, 4, 7, 10]
const DEFAULT_BATCH_SIZE = 100
const DEFAULT_MIN_H_RESOLUTION = 1000.0f0
const DEFAULT_VERBOSE = false
const DEFAULT_PAUSE_BETWEEN_BATCHES = false
const DEFAULT_FORCE_REPROCESS = false
const DEFAULT_TIMESTEP_BATCH_SIZE = 0
const SCRIPT_DIR = @__DIR__
const BATCH_SCRIPT_PATH = @__FILE__

const PREFERRED_PARALLEL_BACKEND = :serial
const PREFERRED_BATCH_SIZE = 100
const PREFERRED_TIMESTEP_BATCH_SIZE = 0
const PREFERRED_VERBOSE = true

function case_arrow_filename(site_id::Int, month::Int, experiment::String)
    site_id_str = string(site_id, pad=3)
    month_str = string(month, pad=2)
    return "googleles_case__$(site_id_str)__month__$(month_str)__exp__$(experiment).arrow"
end

function case_arrow_path(site_id::Int, month::Int, experiment::String, output_dir::String)
    return joinpath(output_dir, case_arrow_filename(site_id, month, experiment))
end

function count_rows(file_path::String)
    if !isfile(file_path)
        return 0
    end
    return DataFrames.nrow(DataFrames.DataFrame(Arrow.Table(file_path)))
end

function case_rows(site_id::Int, month::Int, experiment::String, output_dir::String)
    return count_rows(case_arrow_path(site_id, month, experiment, output_dir))
end

function case_is_satisfied(case::NamedTuple; checked_cases, experiment::String, output_dir::String)
    arrow_path = case_arrow_path(case.site_id, case.month, experiment, output_dir)
    # Canonical condition: Arrow file exists (non-empty for generated, zero-row for cloudfree).
    return isfile(arrow_path)
end

function expand_cases(candidate_sites::Vector{Int}, candidate_months::Vector{Int})
    cases = NamedTuple{(:site_id, :month)}[]
    for month in candidate_months
        for site_id in candidate_sites
            push!(cases, (site_id=site_id, month=month))
        end
    end
    return cases
end

function prompt_continue(batch_index::Int, total_batches::Int)
    if !(stdin isa Base.TTY) && !isinteractive()
        return true
    end

    print("Finished batch $(batch_index)/$(total_batches). Continue? [Y/n] ")
    flush(stdout)
    answer = lowercase(strip(readline(stdin)))
    return isempty(answer) || answer in ("y", "yes")
end

function run_case!(site_id::Int, month::Int; experiment::String, output_dir::String, max_timesteps::Int, timestep_batch_size::Int, min_h_resolution::Float32, include_cfsites::Bool, verbose::Bool, tabular_options::MLCD.TabularBuildOptions)
    started_at = time()
    println("Case start: site=$(site_id), month=$(month), experiment=$(experiment), max_timesteps=$(max_timesteps), timestep_batch_size=$(timestep_batch_size)")

    generate_data!(;
        site_id=site_id,
        month=month,
        experiment=experiment,
        output_dir=output_dir,
        max_timesteps=max_timesteps,
        timestep_batch_size=timestep_batch_size,
        min_h_resolution=min_h_resolution,
        include_cfsites=include_cfsites,
        verbose=verbose,
        tabular_options=tabular_options,
    )

    rows = case_rows(site_id, month, experiment, output_dir)
    status = rows > 0 ? "generated" : "cloudfree"
    elapsed = round(time() - started_at; digits=1)
    println("Case done: site=$(site_id), month=$(month), status=$(status), rows=$(rows), elapsed=$(elapsed)s")
    println("\n----------------\n")
    return (site_id=site_id, month=month, rows=rows, status=status, timestamp=string(Dates.now()))
end

function run_case_batch(cases::AbstractVector{<:NamedTuple}; experiment::String, output_dir::String, max_timesteps::Int, timestep_batch_size::Int, min_h_resolution::Float32, include_cfsites::Bool, verbose::Bool, threaded::Bool, tabular_options::MLCD.TabularBuildOptions)
    if threaded
        tasks = [Base.Threads.@spawn run_case!(case.site_id, case.month; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, tabular_options=tabular_options) for case in cases]
        return fetch.(tasks)
    end

    results = NamedTuple[]
    for case in cases
        push!(results, run_case!(case.site_id, case.month; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, tabular_options=tabular_options))
    end
    return results
end

function run_case_batch_omt(cases::AbstractVector{<:NamedTuple}; experiment::String, output_dir::String, max_timesteps::Int, timestep_batch_size::Int, min_h_resolution::Float32, include_cfsites::Bool, verbose::Bool, tabular_options::MLCD.TabularBuildOptions)
    # Keep OMT optional at runtime so the experiment can stay lean.
    pkg_path = Base.find_package("OhMyThreads")
    if pkg_path === nothing
        @warn "PARALLEL_BACKEND=omt requested, but OhMyThreads is not available in this environment; falling back to Base threads."
        return run_case_batch(cases; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, threaded=true, tabular_options=tabular_options)
    end

    OMT = Base.require(Base.PkgId(Base.UUID("67456a42-1dca-4109-a031-0a68de7e3ad5"), "OhMyThreads"))
    return OMT.tmap(cases) do case
        run_case!(case.site_id, case.month; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, tabular_options=tabular_options)
    end
end

function run_case_batch_distributed(cases::AbstractVector{<:NamedTuple}; experiment::String, output_dir::String, max_timesteps::Int, timestep_batch_size::Int, min_h_resolution::Float32, include_cfsites::Bool, verbose::Bool, tabular_options::MLCD.TabularBuildOptions)
    if Distributed.nworkers() == 1
        @warn "Distributed backend requested but no workers are available; falling back to Base threads."
        return run_case_batch(cases; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, threaded=true, tabular_options=tabular_options)
    end

    for w in Distributed.workers()
        Distributed.remotecall_wait(w, SCRIPT_DIR, BATCH_SCRIPT_PATH) do script_dir, batch_script_path
            if !isdefined(Main, :run_case!)
                Base.include(Main, batch_script_path)
            end
            nothing
        end
    end

    return Distributed.pmap(cases) do case
        Main.Random.seed!(1234 + case.site_id + 1000 * case.month)
        Main.run_case!(case.site_id, case.month; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, tabular_options=tabular_options)
    end
end

function batch_generate_data!(;
    candidate_sites::Vector{Int} = MLCD.EnvHelpers.parse_int_list_env("CANDIDATE_SITES", DEFAULT_CANDIDATE_SITES),
    candidate_months::Vector{Int} = MLCD.EnvHelpers.parse_int_list_env("CANDIDATE_MONTHS", DEFAULT_CANDIDATE_MONTHS),
    batch_size::Int = parse(Int, get(ENV, "BATCH_SIZE", string(DEFAULT_BATCH_SIZE))),
    experiment::String = "amip",
    output_dir::String = MLCD.Paths.processed_data_root(),
    max_timesteps::Int = parse(Int, get(ENV, "MAX_TIMESTEPS", "0")),
    timestep_batch_size::Int = parse(Int, get(ENV, "TIMESTEP_BATCH_SIZE", string(DEFAULT_TIMESTEP_BATCH_SIZE))),
    min_h_resolution::Float32 = parse(Float32, get(ENV, "MIN_H_RESOLUTION", string(DEFAULT_MIN_H_RESOLUTION))),
    include_cfsites::Bool = MLCD.EnvHelpers.parse_bool_env("INCLUDE_CFSITES", false),
    verbose::Bool = MLCD.EnvHelpers.parse_bool_env("VERBOSE_GENERATION", DEFAULT_VERBOSE),
    parallel_backend::Symbol = Symbol(lowercase(get(ENV, "PARALLEL_BACKEND", MLCD.EnvHelpers.parse_bool_env("THREADED", true) ? "threads" : "serial"))),
    pause_between_batches::Bool = MLCD.EnvHelpers.parse_bool_env("PAUSE_BETWEEN_BATCHES", DEFAULT_PAUSE_BETWEEN_BATCHES),
    force_reprocess::Bool = MLCD.EnvHelpers.parse_bool_env("FORCE_REPROCESS", DEFAULT_FORCE_REPROCESS),
    tabular_options::MLCD.TabularBuildOptions = MLCD.tabular_build_options_from_env(),
)
    mkpath(output_dir)
    mkpath(MLCD.Paths.workflow_cache_root())

    cases = expand_cases(candidate_sites, candidate_months)
    checked_cases = MLCD.WorkflowState.load_checked_cases(MLCD.WorkflowState.checked_cases_path("amip_baseline"))
    pending_cases = if force_reprocess
        cases
    else
        filter(case -> !case_is_satisfied(case; checked_cases=checked_cases, experiment=experiment, output_dir=output_dir), cases)
    end

    if isempty(pending_cases)
        println("No unprocessed cases found; nothing to do.")
        return output_dir
    end

    total_batches = cld(length(pending_cases), max(batch_size, 1))
    println("Processing $(length(pending_cases)) cases in $(total_batches) batches of up to $(batch_size) case(s) each.")
    println("Parallel backend: $(parallel_backend == :distributed ? "distributed" : parallel_backend == :omt ? "omt" : parallel_backend == :threads ? "threads" : "serial")")
    println("Pause between batches: $(pause_between_batches)")
    println("Force reprocess: $(force_reprocess)")
    println("Timestep batch size: $(timestep_batch_size)")
    if verbose
        println(MLCD.tabular_build_options_summary(tabular_options))
    end

    cache_path = MLCD.WorkflowState.checked_cases_path("amip_baseline")
    processed_rows = 0

    for (batch_index, start_idx) in enumerate(1:max(batch_size, 1):length(pending_cases))
        stop_idx = min(start_idx + batch_size - 1, length(pending_cases))
        batch_cases = pending_cases[start_idx:stop_idx]
        println("Starting batch $(batch_index)/$(total_batches) with $(length(batch_cases)) case(s)...")

        results = if parallel_backend == :distributed
            run_case_batch_distributed(batch_cases; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, tabular_options=tabular_options)
        elseif parallel_backend == :omt
            run_case_batch_omt(batch_cases; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, tabular_options=tabular_options)
        else
            run_case_batch(batch_cases; experiment=experiment, output_dir=output_dir, max_timesteps=max_timesteps, timestep_batch_size=timestep_batch_size, min_h_resolution=min_h_resolution, include_cfsites=include_cfsites, verbose=verbose, threaded=(parallel_backend == :threads), tabular_options=tabular_options)
        end

        batch_rows = 0
        for result in results
            key = MLCD.WorkflowState.case_key(result.site_id, result.month, experiment)
            MLCD.WorkflowState.record_checked_case!(cache_path, result.site_id, result.month, experiment; status=result.status, rows=result.rows, timesteps=max_timesteps)
            checked_cases[key] = (status = result.status, rows = result.rows, timesteps = max_timesteps, timestamp = result.timestamp)
            batch_rows += result.rows
        end

        processed_rows += batch_rows
        println("Finished batch $(batch_index)/$(total_batches); rows added this batch=$(batch_rows), rows added total=$(processed_rows)")

        if batch_index < total_batches && pause_between_batches && !prompt_continue(batch_index, total_batches)
            println("Stopping after batch $(batch_index) at user request.")
            break
        end
    end

    println("Batch data generation complete: $(output_dir)")
    return output_dir
end

"""
    run_preferred_batch_generate!(; kwargs...)

Preferred single-entry workflow for interactive REPL use.
Defaults to full-case GoogleLES loading, serial batch execution, and
no pause between batches. Override any keyword argument as needed.
"""
function run_preferred_batch_generate!(;
    candidate_sites::Vector{Int} = MLCD.EnvHelpers.parse_int_list_env("CANDIDATE_SITES", DEFAULT_CANDIDATE_SITES),
    candidate_months::Vector{Int} = MLCD.EnvHelpers.parse_int_list_env("CANDIDATE_MONTHS", DEFAULT_CANDIDATE_MONTHS),
    batch_size::Int = PREFERRED_BATCH_SIZE,
    max_timesteps::Int = parse(Int, get(ENV, "MAX_TIMESTEPS", "0")),
    force_reprocess::Bool = MLCD.EnvHelpers.parse_bool_env("FORCE_REPROCESS", DEFAULT_FORCE_REPROCESS),
)
    tabular_opts = MLCD.tabular_build_options_from_env(force_fullcase=true)

    # use threaded if threads are available, otherwise fall back to serial; this is just for convenience in interactive use, not a hard requirement
    if Threads.nthreads() > 1
        parallel_backend = :threads
    else
        parallel_backend = :serial
    end

    return batch_generate_data!(;
        candidate_sites=candidate_sites,
        candidate_months=candidate_months,
        batch_size=batch_size,
        max_timesteps=max_timesteps,
        timestep_batch_size=PREFERRED_TIMESTEP_BATCH_SIZE,
        parallel_backend=parallel_backend,
        pause_between_batches=false,
        force_reprocess=force_reprocess,
        verbose=PREFERRED_VERBOSE,
        tabular_options=tabular_opts,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_preferred_batch_generate!()
end