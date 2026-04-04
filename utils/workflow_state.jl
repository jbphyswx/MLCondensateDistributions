"""
Persistence helpers for workflow bookkeeping.

This stores small TSV manifests under the package-wide data directory so
repeated bootstrap runs can skip simulations that were already checked.
"""
module WorkflowState

using Dates: Dates

include("paths.jl")
using .Paths

export checked_cases_path, case_key, load_checked_cases, record_checked_case!, was_checked

"""Return the TSV manifest path for a named workflow."""
checked_cases_path(workflow::AbstractString="amip_baseline") = joinpath(Paths.workflow_cache_root(), "$(workflow)_checked.tsv")

"""Build the stable key used to identify a site/month/experiment case."""
case_key(site_id::Integer, month::Integer, experiment::AbstractString) = "$(site_id)|$(month)|$(experiment)"

"""Load the last recorded status for each checked case."""
function load_checked_cases(path::AbstractString)
    records = Dict{String, NamedTuple{(:status, :rows, :timesteps, :timestamp), Tuple{String, Int, Int, String}}}()
    if !isfile(path)
        return records
    end

    for line in eachline(path)
        isempty(strip(line)) && continue
        startswith(line, "#") && continue
        parts = split(line, '\t')
        length(parts) < 5 && continue
        key = parts[1]
        status = parts[2]
        rows = tryparse(Int, parts[3])
        timesteps = tryparse(Int, parts[4])
        timestamp = parts[5]
        records[key] = (status = status, rows = something(rows, 0), timesteps = something(timesteps, 0), timestamp = timestamp)
    end

    return records
end

"""Return whether the case already appears in the manifest."""
was_checked(path::AbstractString, key::AbstractString) = haskey(load_checked_cases(path), key)

"""Append or update the checked-case manifest with the latest status."""
function record_checked_case!(path::AbstractString, site_id::Integer, month::Integer, experiment::AbstractString; status::AbstractString, rows::Integer=0, timesteps::Integer=0)
    mkpath(dirname(path))
    open(path, "a") do io
        println(io, "$(case_key(site_id, month, experiment))\t$(status)\t$(rows)\t$(timesteps)\t$(Dates.now())")
    end
end

end # module