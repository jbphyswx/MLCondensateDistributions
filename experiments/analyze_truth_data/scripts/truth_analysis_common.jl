using DataFrames: DataFrames

"""Keep only covariance moment columns (names starting with `cov_`)."""
function covariance_targets(targets::Vector{Symbol})
    return [t for t in targets if startswith(String(t), "cov_")]
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
