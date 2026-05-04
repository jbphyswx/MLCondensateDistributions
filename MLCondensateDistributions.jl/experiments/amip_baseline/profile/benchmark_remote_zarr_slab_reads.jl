#!/usr/bin/env julia
# Manual experiment only (not `Pkg.test`): times remote GCS reads for one GoogleLES field using the
# same `_load_googleles_timestep_fields!` path as `build_tabular`.
#
# `_load_googleles_timestep_fields!` only installs **lazy views**; bytes move on materialization
# (`copyto!`). This script times **load + copyto!** into a dense `Array{Float32}` (like the pipeline).
#
# Requires network. Compares full-z vs contiguous z-range vs fused vector z-indices for one timestep.
# Shapes come from lazy metadata (no full-column probe). Timed samples interleave modes in random order each round.
#
# When the fused `idx` has ≥2 contiguous runs (e.g. 10:24 and 80:94), also times **two sequential**
# `UnitRange` loads (same total z levels as one fused load) — mimics multiple span-group passes without fusion.
#
# Usage (from repo root):
#   julia --project=. experiments/amip_baseline/profile/benchmark_remote_zarr_slab_reads.jl [site] [month] [t_1based] [experiment]

const ROOT = dirname(dirname(dirname(@__DIR__)))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD
using Random: Random
using Statistics: Statistics
const FIELD_SPECS = (("T", "ta"),)
const N_SAMPLES = 7
const N_WARMUP_ROUNDS = 4

"""Contiguous `UnitRange`s covering a sorted list of z indices (no duplicates)."""
function _z_contiguous_runs(sorted_z::Vector{Int})::Vector{UnitRange{Int}}
    isempty(sorted_z) && return UnitRange{Int}[]
    runs = UnitRange{Int}[]
    a = sorted_z[1]
    b = sorted_z[1]
    for k in 2:length(sorted_z)
        if sorted_z[k] == b + 1
            b = sorted_z[k]
        else
            push!(runs, a:b)
            a = sorted_z[k]
            b = sorted_z[k]
        end
    end
    push!(runs, a:b)
    return runs
end

"""
    _benchmark_ta_nx_ny_nz(ds, t1) -> (nx, ny, nz)

Lazy `(x,y,z)` shape for timestep `t1` from Zarr metadata + views only — **no full-column read**,
so we do not warm the object-store cache before timed runs.
"""
function _benchmark_ta_nx_ny_nz(ds, t1::Int)
    var = ds["T"]
    haskey(var.attrs, "_ARRAY_DIMENSIONS") || error("GoogleLES variable 'T' is missing _ARRAY_DIMENSIONS")
    metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
    length(metadata_dim_names_raw) == 4 || error("expected 4 dims on T")
    julia_dim_names = (
        Symbol(metadata_dim_names_raw[4]),
        Symbol(metadata_dim_names_raw[3]),
        Symbol(metadata_dim_names_raw[2]),
        Symbol(metadata_dim_names_raw[1]),
    )
    t_axis_idx = MLCD._find_dim_idx(julia_dim_names, :t)
    t_axis_idx == 0 && error("T missing dim t")
    raw = MLCD._slice_t_range_4d(var, t1:t1, t_axis_idx)
    canonical = MLCD._reorder_to_txyz_view(raw, julia_dim_names)
    sz = size(@view(canonical[1, :, :, :]))
    return (sz[1], sz[2], sz[3])
end

"""Load views, then force the same remote I/O the tabular path relies on (`copyto!` into dense storage)."""
function _materialize_ta!(cache, ds, t1, z_subset, dest::Array{Float32,3})
    MLCD._load_googleles_timestep_fields!(cache, ds, t1; field_specs=FIELD_SPECS, z_subset=z_subset)
    a = cache["ta"]
    @assert size(dest) == size(a)
    copyto!(dest, a)
    return nothing
end

function main()
    site = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 313
    month = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    t1 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    experiment = length(ARGS) >= 4 ? ARGS[4] : "amip"

    ds = MLCD.GoogleLES.load_zarr_simulation(site, month, experiment)
    ds === nothing && error("Failed to open Zarr (network or path).")

    cache = Dict{String, AbstractArray{Float32, 3}}()

    nx, ny, nz = _benchmark_ta_nx_ny_nz(ds, t1)
    dest_full = Array{Float32}(undef, nx, ny, nz)

    z_hi = min(40, nz)
    idx = if nz >= 94
        vcat(collect(10:24), collect(80:94))
    else
        mid = max(1, nz ÷ 2)
        vcat(collect(1:min(5, mid)), collect(max(mid + 1, nz - 3):nz))
    end

    z_runs = _z_contiguous_runs(sort!(unique(idx)))
    multi_run = length(z_runs) >= 2
    dest_per_run = multi_run ? [Array{Float32}(undef, nx, ny, length(r)) for r in z_runs] : Array{Float32, 3}[]

    println("== benchmark_remote_zarr_slab_reads: site=$site month=$month t=$t1 experiment=$(repr(experiment)) ==")
    println("  field=T→ta  nz=$nz  contiguous_subset=1:$z_hi  vector_idx_len=$(length(idx))")
    if multi_run
        println("  same z set as $(length(z_runs)) contiguous runs: $(z_runs)  (two-pass vs fused-vector test)")
    else
        println("  fused idx is a single contiguous run — skipping two-pass vs fused comparison (would be identical pattern).")
    end

    dest_sub = Array{Float32}(undef, nx, ny, z_hi)
    dest_vec = Array{Float32}(undef, nx, ny, length(idx))

    active = Symbol[:full, :sub, :vec]
    multi_run && push!(active, :twopass)

    function run_mode!(mode::Symbol)
        if mode === :full
            _materialize_ta!(cache, ds, t1, nothing, dest_full)
        elseif mode === :sub
            _materialize_ta!(cache, ds, t1, 1:z_hi, dest_sub)
        elseif mode === :vec
            _materialize_ta!(cache, ds, t1, idx, dest_vec)
        elseif mode === :twopass
            for (r, dest_r) in zip(z_runs, dest_per_run)
                _materialize_ta!(cache, ds, t1, r, dest_r)
            end
        else
            error("unknown mode $(repr(mode))")
        end
    end

    # Compile + touch Zarr once with a single z-level (minimal bytes vs full-column probe).
    dest_compile = Array{Float32}(undef, nx, ny, 1)
    _materialize_ta!(cache, ds, t1, 1:1, dest_compile)

    samples = Dict{Symbol, Vector{Float64}}(m => Float64[] for m in active)
    for _ in 1:N_WARMUP_ROUNDS
        for mode in Random.shuffle(active)
            run_mode!(mode)
        end
    end
    for _ in 1:N_SAMPLES
        for mode in Random.shuffle(active)
            push!(samples[mode], @elapsed run_mode!(mode))
        end
    end

    function _summarize(samples_v)
        Statistics.median(samples_v), samples_v
    end

    t_full, s_full = _summarize(samples[:full])
    t_sub, s_sub = _summarize(samples[:sub])
    t_vec, s_vec = _summarize(samples[:vec])
    t_2pass, s_2pass = if multi_run
        _summarize(samples[:twopass])
    else
        (NaN, Float64[])
    end

    _fmt_ms(t) = string(round(t * 1000; digits=2), " ms")
    println("\nMedian wall time — load + copyto! (dense); $(N_WARMUP_ROUNDS) interleaved warmup rounds, $(N_SAMPLES) timed rounds × $(length(active)) shuffled modes/round:")
    println("  full_z (all levels)      median=$( _fmt_ms(t_full))  (min…max ms: $(round(Statistics.minimum(s_full)*1000;digits=2)) … $(round(Statistics.maximum(s_full)*1000;digits=2)))")
    println("  contiguous z 1:$z_hi      median=$( _fmt_ms(t_sub))  (min…max: $(round(Statistics.minimum(s_sub)*1000;digits=2)) … $(round(Statistics.maximum(s_sub)*1000;digits=2)))")
    println("  fused vector z (len=$(length(idx))) median=$( _fmt_ms(t_vec))  (min…max: $(round(Statistics.minimum(s_vec)*1000;digits=2)) … $(round(Statistics.maximum(s_vec)*1000;digits=2)))")
    if multi_run
        println("  $(length(z_runs)) sequential UnitRange loads (same z levels) median=$( _fmt_ms(t_2pass))  (min…max: $(round(Statistics.minimum(s_2pass)*1000;digits=2)) … $(round(Statistics.maximum(s_2pass)*1000;digits=2)))")
        r = t_2pass / t_vec
        println("    ratio 2pass/fused = $(round(r; digits=3))  (<1 ⇒ two UnitRange passes faster; >1 ⇒ one fused vector read faster)")
    end
    println("\nNo full-z probe before timing; modes run in random order each round so cache is shared fairly across comparisons. For a cold-cache study, restart Julia between runs or use a fresh store URL.")
end

main()
