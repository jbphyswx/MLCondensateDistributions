"""
Generic data-loading helpers shared across training and analysis workflows.
"""
module DataHandling

using DataFrames: DataFrames
using Parquet2: Parquet2

export list_parquet_files, select_columns, load_parquet_dataframe, validate_required_columns
export load_moments_dataframe

"""Return sorted parquet files from `data_dir` (optionally limited by `max_files`)."""
function list_parquet_files(data_dir::String; max_files::Int = 0)
    files = filter(f -> endswith(f, ".parquet"), readdir(data_dir; join=true))
    isempty(files) && error("No .parquet files found in $(data_dir)")
    files = sort(files)
    if max_files > 0
        return files[1:min(max_files, length(files))]
    end
    return files
end

"""Validate DataFrame contains all required columns; throws if missing."""
function validate_required_columns(df::DataFrames.DataFrame, required_cols::Vector{Symbol})
    cols = Set(Symbol.(names(df)))
    missing = [c for c in required_cols if !(c in cols)]
    !isempty(missing) && error("Missing required columns: $(missing)")
    return true
end

"""
Select columns from `colnames` by explicit names, prefix list, and/or regex.

Returns sorted unique symbols.
"""
function select_columns(
    colnames::Vector{Symbol};
    names::Vector{Symbol} = Symbol[],
    prefixes::Vector{String} = String[],
    regex::Union{Regex, Nothing} = nothing,
)
    out = Set{Symbol}(names)
    for c in colnames
        s = String(c)
        if any(p -> startswith(s, p), prefixes)
            push!(out, c)
        end
        if regex !== nothing && occursin(regex, s)
            push!(out, c)
        end
    end
    return sort!(collect(out); by=String)
end

"""
Load and concatenate parquet files from `data_dir`, selecting only `columns`.

When `drop_empty=true`, zero-row files are skipped.
"""
function load_parquet_dataframe(
    data_dir::String;
    columns::Vector{Symbol} = Symbol[],
    max_files::Int = 0,
    drop_empty::Bool = true,
)
    files = list_parquet_files(data_dir; max_files=max_files)
    dfs = DataFrames.DataFrame[]
    for f in files
        df = DataFrames.DataFrame(Parquet2.Dataset(f))
        if drop_empty && DataFrames.nrow(df) == 0
            continue
        end
        if !isempty(columns)
            validate_required_columns(df, columns)
            push!(dfs, DataFrames.select(df, columns))
        else
            push!(dfs, df)
        end
    end
    isempty(dfs) && error("No non-empty parquet files found in $(data_dir)")
    return vcat(dfs...)
end

"""
    load_moments_dataframe(data_dir; max_files=0, target_prefixes=[...], extra_columns=Symbol[])

Load processed parquet files with horizontal/vertical resolution, all moment columns matching
`target_prefixes`, and any `extra_columns` (e.g. `:tke`, `:liq_fraction`).

Returns `(df, target_cols)` where `target_cols` are only the moment columns (not `extra_columns`).
"""
function load_moments_dataframe(
    data_dir::String;
    max_files::Int = 0,
    target_prefixes::Vector{String} = ["cov_", "var_"],
    extra_columns::Vector{Symbol} = Symbol[],
)
    probe_files = list_parquet_files(data_dir; max_files=1)
    schema_df = DataFrames.DataFrame(Parquet2.Dataset(first(probe_files)))
    colnames = Symbol.(names(schema_df))
    target_cols = select_columns(colnames; prefixes=target_prefixes)

    required_cols = unique!(vcat([:resolution_h, :resolution_z], extra_columns, target_cols))
    df = load_parquet_dataframe(data_dir; columns=required_cols, max_files=max_files, drop_empty=true)
    return df, target_cols
end

end # module
