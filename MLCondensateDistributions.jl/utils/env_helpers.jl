"""
Generic environment parsing helpers shared across scripts.
"""
module EnvHelpers

export parse_bool_env, parse_int_list_env

"""Parse environment variable as Bool with fallback `default`."""
function parse_bool_env(name::String, default::Bool)
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default
    return lowercase(raw) in ("1", "true", "yes", "y", "on")
end

"""Parse comma-separated integer list from environment with fallback defaults."""
function parse_int_list_env(name::String, default_values::Vector{Int}) # Todo, support tuples, ranges, etc.. parsing from string hampers returning type stable types though so maybe we're stuck with vector outs . would be good to cast ouotput a tye stable type in downstream functions perhaps? idk if it's possible though
    raw = strip(get(ENV, name, ""))
    isempty(raw) && return default_values
    out = Int[]
    for part in split(raw, ',')
        p = strip(part)
        isempty(p) && continue
        v = tryparse(Int, p)
        v === nothing && error("Invalid integer in $(name): $(p)")
        push!(out, v)
    end
    isempty(out) ? default_values : out
end

end # module
