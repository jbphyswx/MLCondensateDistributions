using Arrow: Arrow
using DataFrames: DataFrames
using Statistics: Statistics
using Random: Random

"""
    load_processed_data(data_dir::String, feature_cols::Vector{Symbol}, target_cols::Vector{Symbol})

Loads all .arrow files from a directory and returns (features, targets) as matrices.
"""
function load_processed_data(data_dir::String, feature_cols::Vector{Symbol}, target_cols::Vector{Symbol})
    all_files = filter(f -> endswith(f, ".arrow"), readdir(data_dir, join=true))
    
    if isempty(all_files)
        error("No .arrow files found in $data_dir. Please run utils/build_training_data.jl first.")
    end
    
    dfs = []
    for f in all_files
        table = Arrow.Table(f)
        push!(dfs, DataFrames.DataFrame(table))
    end
    
    df_full = vcat(dfs...)
    
    X = Matrix{Float32}(df_full[:, feature_cols])'
    Y = Matrix{Float32}(df_full[:, target_cols])'
    
    return X, Y
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
