using DataFrames: DataFrames

"""Keep only covariance moment columns (names starting with `cov_`)."""
function covariance_targets(targets::Vector{Symbol})
    return [t for t in targets if startswith(String(t), "cov_")]
end

"""
For `cov_qt_ql`-style names, return `(var_qt, var_ql)`. Returns `nothing` if the name does not
match `cov_<a>_<b>` with two single-token suffix parts.
"""
function covariance_column_var_pair(cov_col::Symbol)::Union{Nothing,Tuple{Symbol,Symbol}}
    s = String(cov_col)
    startswith(s, "cov_") || return nothing
    rest = s[5:end]
    parts = split(rest, '_')
    length(parts) == 2 || return nothing
    a, b = parts
    (isempty(a) || isempty(b)) && return nothing
    return (Symbol("var_", a), Symbol("var_", b))
end

"""
Add columns `corr_*` (Pearson r = cov / √(var_i var_j)) for each `cov_*` in `cov_cols` where the
matching `var_*` columns exist. Rows with nonpositive or invalid variance product get `NaN`.

Returns the list of new column symbols (only disparate-variable pairs present in `df`).
"""
function append_pearson_correlation_columns!(
    df::DataFrames.DataFrame,
    cov_cols::Vector{Symbol},
)::Vector{Symbol}
    out = Symbol[]
    ε = 1.0e-30
    for c in cov_cols
        p = covariance_column_var_pair(c)
        p === nothing && continue
        va, vb = p
        (hasproperty(df, va) && hasproperty(df, vb)) || continue
        cr = Symbol(replace(String(c), "cov_" => "corr_"; count = 1))
        v1 = Float64.(df[!, va])
        v2 = Float64.(df[!, vb])
        cv = Float64.(df[!, c])
        prod = v1 .* v2
        denom = sqrt.(max.(prod, 0.0))
        invalid = (denom .< sqrt(ε)) .| .!(isfinite.(denom)) .| .!(isfinite.(cv))
        r = cv ./ denom
        r[invalid] .= NaN
        r .= clamp.(r, -1.0, 1.0)
        df[!, cr] = Float32.(r)
        push!(out, cr)
    end
    return out
end

"""
Subset `df` when optional provenance filters are set (string equality on metadata columns).
"""
function apply_provenance_filters(
    df::DataFrames.DataFrame;
    data_source::Union{Nothing,String} = nothing,
    forcing_model::Union{Nothing,String} = nothing,
    experiment::Union{Nothing,String} = nothing,
)
    out = df
    if data_source !== nothing
        out = DataFrames.subset(out, :data_source => DataFrames.ByRow(==(data_source)))
    end
    if forcing_model !== nothing
        out = DataFrames.subset(out, :forcing_model => DataFrames.ByRow(==(forcing_model)))
    end
    if experiment !== nothing
        out = DataFrames.subset(out, :experiment => DataFrames.ByRow(==(experiment)))
    end
    return out
end
