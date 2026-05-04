using Arrow: Arrow
using DataFrames: DataFrames
using Parquet2: Parquet2

"""
Convert all .arrow files under `data_dir` into .parquet files with Snappy compression.

The script enforces one-way migration semantics:
- It errors if any converted parquet file already exists.
- It validates row count and column order equality for each converted file.
- It deletes the source .arrow file only after successful validation.
"""
function migrate_arrow_to_parquet!(data_dir::String)
    isdir(data_dir) || error("Data directory does not exist: $(data_dir)")

    arrow_files = sort(filter(f -> endswith(f, ".arrow"), readdir(data_dir; join=true)))
    isempty(arrow_files) && error("No .arrow files found in $(data_dir)")

    println("Found $(length(arrow_files)) arrow files in $(data_dir)")

    for arrow_path in arrow_files
        parquet_path = replace(arrow_path, r"\.arrow$" => ".parquet")
        isfile(parquet_path) && error("Refusing to overwrite existing parquet file: $(parquet_path)")

        df_arrow = DataFrames.DataFrame(Arrow.Table(arrow_path))
        Parquet2.writefile(parquet_path, df_arrow; compression_codec=:snappy)
        df_parquet = DataFrames.DataFrame(Parquet2.Dataset(parquet_path))

        nrow(df_arrow) == nrow(df_parquet) || error("Row mismatch after conversion for $(arrow_path)")
        names(df_arrow) == names(df_parquet) || error("Column mismatch after conversion for $(arrow_path)")

        rm(arrow_path; force=true)
        println("Converted and removed: $(arrow_path) -> $(parquet_path)")
    end

    println("Arrow to parquet migration complete.")
    return nothing
end

function main()
    repo_root = normpath(joinpath(@__DIR__, ".."))
    data_dir = joinpath(repo_root, "data", "processed")
    migrate_arrow_to_parquet!(data_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
