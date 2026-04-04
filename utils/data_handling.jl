"""
Generic data-loading helpers shared across training and analysis workflows.
"""
module DataHandling

using Arrow: Arrow
using DataFrames: DataFrames

export list_arrow_files, select_columns, load_arrow_dataframe, validate_required_columns
export collect_arrow_files, select_columns_by_prefix, load_resolution_moments_df

"""Return sorted Arrow files from `data_dir` (optionally limited by `max_files`)."""
function list_arrow_files(data_dir::String; max_files::Int = 0)
    files = filter(f -> endswith(f, ".arrow"), readdir(data_dir; join=true))
    isempty(files) && error("No .arrow files found in $(data_dir)")
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
Load and concatenate Arrow files from `data_dir`, selecting only `columns`.

When `drop_empty=true`, zero-row files are skipped.
"""
function load_arrow_dataframe(
    data_dir::String;
    columns::Vector{Symbol} = Symbol[],
    max_files::Int = 0,
    drop_empty::Bool = true,
)
    files = list_arrow_files(data_dir; max_files=max_files)
    dfs = DataFrames.DataFrame[]
    for f in files
        df = DataFrames.DataFrame(Arrow.Table(f))
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
    isempty(dfs) && error("No non-empty Arrow files found in $(data_dir)")
    return vcat(dfs...)
end

# Compatibility wrappers (temporary)
"""Deprecated compatibility wrapper. Use `list_arrow_files` instead."""
function collect_arrow_files(data_dir::String; max_files::Int = 0)
    Base.depwarn("collect_arrow_files is deprecated; use list_arrow_files", :collect_arrow_files)
    return list_arrow_files(data_dir; max_files=max_files)
end

"""Deprecated compatibility wrapper. Use `select_columns(...; prefixes=...)` instead."""
function select_columns_by_prefix(colnames::Vector{Symbol}, prefixes::Vector{String})
    Base.depwarn("select_columns_by_prefix is deprecated; use select_columns(...; prefixes=...)", :select_columns_by_prefix)
    return select_columns(colnames; prefixes=prefixes)
end

"""
Load processed Arrow files and return a DataFrame with resolution + moment columns.

Returns `(df, target_cols)` where `target_cols` are all columns matching
`target_prefixes`.
"""
function load_resolution_moments_df(
    data_dir::String;
    max_files::Int = 0,
    target_prefixes::Vector{String} = ["cov_", "var_"],
)
    Base.depwarn("load_resolution_moments_df is transitional; prefer load_arrow_dataframe + select_columns", :load_resolution_moments_df)

    files = list_arrow_files(data_dir; max_files=max_files)
    first_df = DataFrames.DataFrame(Arrow.Table(first(files)))
    cols = Symbol.(names(first_df))
    target_cols = select_columns(cols; prefixes=target_prefixes)

    required_cols = vcat([:resolution_h, :resolution_z], target_cols)
    df = load_arrow_dataframe(data_dir; columns=required_cols, max_files=max_files, drop_empty=true)
    return df, target_cols
end

end # module
