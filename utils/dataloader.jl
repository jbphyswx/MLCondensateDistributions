using Arrow: Arrow
using DataFrames: DataFrames
using Statistics: Statistics
using Random: Random

"""
    prune_incompatible_arrow_files!(data_dir::String, required_cols::Vector{Symbol}; verbose::Bool=true)

Delete Arrow files in `data_dir` that are empty or do not contain all `required_cols`.
Returns a vector of compatible file paths.
"""
function prune_incompatible_arrow_files!(data_dir::String, required_cols::Vector{Symbol}; verbose::Bool=true)
    all_files = filter(f -> endswith(f, ".arrow"), readdir(data_dir, join=true))
    compatible_files = String[]

    for f in all_files
        table = Arrow.Table(f)
        df = DataFrames.DataFrame(table)

        if DataFrames.ncol(df) == 0 || DataFrames.nrow(df) == 0
            rm(f; force=true)
            if verbose
                @warn "Removed empty Arrow file" file=f
            end
            continue
        end

        file_cols = Set(Symbol.(names(df)))
        if !all(c -> c in file_cols, required_cols)
            missing_cols = [c for c in required_cols if !(c in file_cols)]
            rm(f; force=true)
            if verbose
                @warn "Removed incompatible Arrow file with missing training columns" file=f missing=missing_cols
            end
            continue
        end

        has_nonfinite = false
        for nm in names(df)
            col = df[!, nm]
            if !(eltype(col) <: Real)
                continue
            end
            if any(x -> !isfinite(x), col)
                has_nonfinite = true
                break
            end
        end

        if has_nonfinite
            rm(f; force=true)
            if verbose
                @warn "Removed Arrow file containing non-finite numeric values" file=f
            end
            continue
        end

        push!(compatible_files, f)
    end

    return compatible_files
end

"""
    load_processed_data(data_dir::String, feature_cols::Vector{Symbol}, target_cols::Vector{Symbol})

Loads all .arrow files from a directory and returns (features, targets) as matrices.
"""
function load_processed_data(data_dir::String, feature_cols::Vector{Symbol}, target_cols::Vector{Symbol})
    required_cols = vcat(feature_cols, target_cols)
    compatible_files = prune_incompatible_arrow_files!(data_dir, required_cols)
    
    if isempty(compatible_files)
        error("No .arrow files found in $data_dir. Please run utils/build_training_data.jl first.")
    end
    
    dfs = DataFrames.DataFrame[]
    for f in compatible_files
        table = Arrow.Table(f)
        df = DataFrames.DataFrame(table)
        push!(dfs, df)
    end

    if isempty(dfs)
        error("No compatible non-empty .arrow files in $data_dir. Ensure processed files include all required training columns.")
    end
    
    df_full = vcat(dfs...)

    full_cols = Set(Symbol.(names(df_full)))
    missing_cols = [c for c in required_cols if !(c in full_cols)]
    if !isempty(missing_cols)
        error("Missing required columns in processed dataset: $(missing_cols)")
    end
    
    X = Matrix{Float32}(df_full[:, feature_cols])'
    Y = Matrix{Float32}(df_full[:, target_cols])'
    
    return X, Y
end

"""
    preview_processed_file(file_path::String; n::Int=10)

Print schema, shape, and the first `n` rows from a processed Arrow file.
Returns the loaded DataFrame.
"""
function preview_processed_file(file_path::String; n::Int=10)
    df = DataFrames.DataFrame(Arrow.Table(file_path))
    println("File: $(file_path)")
    println("Rows: $(DataFrames.nrow(df)), Cols: $(DataFrames.ncol(df))")
    println("Columns: $(join(string.(names(df)), ", "))")
    println(first(df, n))
    return df
end

"""
    standardize_data(X::AbstractMatrix)

Compute mean and std for each row (feature) and return (standardized_X, means, stds).
"""
function standardize_data(X::AbstractMatrix)
    means = Statistics.mean(X, dims=2)
    stds = Statistics.std(X, dims=2)
    # Avoid division by zero for constant features
    stds[stds .== 0] .= 1.0f0
    X_std = (X .- means) ./ stds
    return X_std, means, stds
end
