module DatasetBuilderAB

using DataFrames: DataFrames
using ..DatasetBuilder: DatasetBuilder

include("dataset_builder_v2.jl")
using .DatasetBuilderV2: DatasetBuilderV2

export process_chunk_ab, benchmark_process_chunk

@inline function _resolve_new_builder()
    if isdefined(DatasetBuilderV2, :process_abstract_chunk_v2)
        return getproperty(DatasetBuilderV2, :process_abstract_chunk_v2)
    elseif isdefined(DatasetBuilderV2, :process_abstract_chunk_new)
        return getproperty(DatasetBuilderV2, :process_abstract_chunk_new)
    elseif isdefined(DatasetBuilder, :process_abstract_chunk_v2)
        return getproperty(DatasetBuilder, :process_abstract_chunk_v2)
    elseif isdefined(DatasetBuilder, :process_abstract_chunk_new)
        return getproperty(DatasetBuilder, :process_abstract_chunk_new)
    end
    return nothing
end

"""
    process_chunk_ab(fine_fields, metadata, spatial_info; mode=:old, both_output=:old)

Run dataset builder in `:old`, `:new`, or `:both` mode for one chunk.
Returns a named tuple with selected output DataFrame plus timing and row counts.

This lives in a separate file by design so legacy pipeline code stays untouched.
"""
function process_chunk_ab(
    fine_fields::Dict{String, <:AbstractArray},
    metadata::Dict{Symbol, Any},
    spatial_info::Dict{Symbol, Any};
    mode::Symbol=:old,
    both_output::Symbol=:old,
)
    mode in (:old, :new, :both) || throw(ArgumentError("mode must be :old, :new, or :both"))
    both_output in (:old, :new) || throw(ArgumentError("both_output must be :old or :new"))

    old_fn = DatasetBuilder.process_abstract_chunk
    new_fn = _resolve_new_builder()

    if mode != :old && isnothing(new_fn)
        throw(ArgumentError("No new builder entrypoint found. Expected DatasetBuilder.process_abstract_chunk_v2 or process_abstract_chunk_new."))
    end

    if mode == :old
        t0 = time()
        df_old = old_fn(fine_fields, metadata, spatial_info)
        return (df=df_old, old_s=time() - t0, new_s=NaN, old_rows=DataFrames.nrow(df_old), new_rows=-1)
    elseif mode == :new
        t0 = time()
        df_new = new_fn(fine_fields, metadata, spatial_info)
        return (df=df_new, old_s=NaN, new_s=time() - t0, old_rows=-1, new_rows=DataFrames.nrow(df_new))
    else
        t0 = time()
        df_old = old_fn(fine_fields, metadata, spatial_info)
        t_old = time() - t0

        t1 = time()
        df_new = new_fn(fine_fields, metadata, spatial_info)
        t_new = time() - t1

        out_df = both_output == :new ? df_new : df_old
        return (df=out_df, old_s=t_old, new_s=t_new, old_rows=DataFrames.nrow(df_old), new_rows=DataFrames.nrow(df_new))
    end
end

"""
    benchmark_process_chunk(fine_fields, metadata, spatial_info; repeats=3, warmup=1)

Simple micro-benchmark for one chunk comparing old vs new builder implementations.
Returns a DataFrame with per-run timing and row counts.
"""
function benchmark_process_chunk(
    fine_fields::Dict{String, <:AbstractArray},
    metadata::Dict{Symbol, Any},
    spatial_info::Dict{Symbol, Any};
    repeats::Int=3,
    warmup::Int=1,
)
    repeats >= 1 || throw(ArgumentError("repeats must be >= 1"))
    warmup >= 0 || throw(ArgumentError("warmup must be >= 0"))

    rows = NamedTuple{(:run, :old_s, :new_s, :old_rows, :new_rows), Tuple{Int, Float64, Float64, Int, Int}}[]

    for _ in 1:warmup
        process_chunk_ab(fine_fields, metadata, spatial_info; mode=:old)
        process_chunk_ab(fine_fields, metadata, spatial_info; mode=:new)
    end

    for run in 1:repeats
        out = process_chunk_ab(fine_fields, metadata, spatial_info; mode=:both, both_output=:old)
        push!(rows, (run=run, old_s=out.old_s, new_s=out.new_s, old_rows=out.old_rows, new_rows=out.new_rows))
    end

    return DataFrames.DataFrame(rows)
end

end # module
