"""
Generic analysis helpers for grouped reductions and table-to-matrix transforms.

This module is intentionally non-visual: it prepares numeric summaries that
plotting modules can render.
"""
module Analysis

using DataFrames: DataFrames
using Statistics: Statistics

export group_reduce, group_reduce_many, build_heatmap_matrix, quantile_summary

"""Group by `by_cols` and reduce one `value_col` using `reducer` (default median)."""
function group_reduce(
    df::DataFrames.DataFrame,
    by_cols::Vector{Symbol},
    value_col::Symbol;
    reducer::Function = Statistics.median,
    output_col::Symbol = :value,
)
    out = DataFrames.combine(DataFrames.groupby(df, by_cols), value_col => reducer => output_col)
    sort!(out, by_cols)
    return out
end

"""Group by `by_cols` and reduce multiple `value_cols` using `reducer`."""
function group_reduce_many(
    df::DataFrames.DataFrame,
    by_cols::Vector{Symbol},
    value_cols::Vector{Symbol};
    reducer::Function = Statistics.median,
)
    specs = [c => reducer => Symbol(string(c), "_reduced") for c in value_cols]
    out = DataFrames.combine(DataFrames.groupby(df, by_cols), specs...)
    sort!(out, by_cols)
    return out
end

"""
Build a dense matrix from grouped `(x_col, y_col, value_col)` reductions.

Returns `(xs, ys, matrix)` where matrix is indexed as `[y, x]`.

TODO :: Either this should be a generic method like `build_grouped_matrix` or it belongs in Viz or something.
"""
function build_heatmap_matrix(
    df::DataFrames.DataFrame,
    x_col::Symbol,
    y_col::Symbol,
    value_col::Symbol;
    reducer::Function = Statistics.median,
)
    xs = sort(unique(Float64.(df[!, x_col])))
    ys = sort(unique(Float64.(df[!, y_col])))
    mat = fill(Float32(NaN), length(ys), length(xs))

    grouped = DataFrames.combine(
        DataFrames.groupby(df, [x_col, y_col]),
        value_col => reducer => :reduced,
    )

    x_idx = Dict(x => i for (i, x) in enumerate(xs))
    y_idx = Dict(y => i for (i, y) in enumerate(ys))

    for row in eachrow(grouped)
        x = Float64(row[x_col])
        y = Float64(row[y_col])
        mat[y_idx[y], x_idx[x]] = Float32(row.reduced)
    end

    return xs, ys, mat
end

"""Compute quantile summary table for one `value_col` grouped by `by_cols`."""
function quantile_summary(
    df::DataFrames.DataFrame,
    by_cols::Vector{Symbol},
    value_col::Symbol;
    qs::Vector{Float64} = [0.1, 0.5, 0.9],
)
    grouped = DataFrames.groupby(df, by_cols)
    out = DataFrames.combine(grouped) do sdf
        vals = Float64.(sdf[!, value_col])
        row = Dict{Symbol, Any}()
        for q in qs
            key = Symbol("q", replace(string(round(Int, q * 100)), " " => ""))
            row[key] = Statistics.quantile(vals, q)
        end
        return DataFrames.DataFrame(row)
    end
    sort!(out, by_cols)
    return out
end

end # module
