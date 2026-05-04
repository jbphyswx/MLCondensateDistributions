#!/usr/bin/env julia
# Times `_load_googleles_timestep_fields_into_span_list!` as used in `build_tabular`:
#   - **G_separate**: G calls with `orig_spans=[r]` (UnitRange path each time)
#   - **one_fused**: 1 call with `orig_spans=[r1, r2, …]` (fused z_idx when fuse env is on)
#
# Default spans sit in **distinct** storage z-chunks (cz=60): 10:24, 80:94, 150:165, …
# Runs G = 2, 3, 4, 5 (and 6, 7 if nz allows) for F=1 (`ta`) and F=all non-q_c.
#
# Usage (repo root, needs network):
#   julia --project=. experiments/amip_baseline/profile/benchmark_nonqc_span_groups_pipeline.jl [site] [month] [t] [experiment] [g_max]
#   Optional 5th arg `g_max` caps G (default 7, truncated by nz).

const ROOT = dirname(dirname(dirname(@__DIR__)))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD
using Random: Random
using Statistics: Statistics

const N_SAMPLES = 7
const N_WARMUP_ROUNDS = 4

# One narrow span per 60-level z-chunk (native z), for disjoint chunk-merge groups on typical stores.
const DEFAULT_SPANS_PER_CHUNK = [
    10:24,
    80:94,
    150:165,
    220:235,
    290:305,
    310:325,
    380:395,
    430:445,
]

function _ta_nx_ny_nz(ds, t1::Int)
    var = ds["T"]
    metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
    julia_dim_names = (
        Symbol(metadata_dim_names_raw[4]),
        Symbol(metadata_dim_names_raw[3]),
        Symbol(metadata_dim_names_raw[2]),
        Symbol(metadata_dim_names_raw[1]),
    )
    t_axis_idx = MLCD._find_dim_idx(julia_dim_names, :t)
    raw = MLCD._slice_t_range_4d(var, t1:t1, t_axis_idx)
    canonical = MLCD._reorder_to_txyz_view(raw, julia_dim_names)
    sz = size(@view(canonical[1, :, :, :]))
    return (sz[1], sz[2], sz[3])
end

function _spans_for_G(nz::Int, G::Int)::Vector{UnitRange{Int}}
    spans = UnitRange{Int}[]
    for r in DEFAULT_SPANS_PER_CHUNK
        last(r) <= nz || continue
        push!(spans, r)
        length(spans) >= G && return spans
    end
    error("nz=$nz too small for G=$G (need $(G) disjoint test spans within column)")
end

function _non_qc_field_specs()
    return Tuple((g, c) for (g, c) in MLCD.GOOGLELES_FIELD_SPECS if c != "q_c")
end

function _alloc_dest(field_specs, nx, ny, nz)
    d = Dict{String, Array{Float32, 3}}()
    for (_, c_var) in field_specs
        d[c_var] = Array{Float32}(undef, nx, ny, nz)
    end
    return d
end

function _run_G_separate!(dest, scratch, ds, t1, field_specs, spans::Vector{UnitRange{Int}})
    for r in spans
        MLCD._load_googleles_timestep_fields_into_span_list!(
            dest, ds, t1; field_specs=field_specs, orig_spans=[r], scratch=scratch,
        )
    end
    return nothing
end

function _run_one_fused!(dest, scratch, ds, t1, field_specs, spans::Vector{UnitRange{Int}})
    MLCD._load_googleles_timestep_fields_into_span_list!(
        dest, ds, t1; field_specs=field_specs, orig_spans=spans, scratch=scratch,
    )
    return nothing
end

function _bench_G(
    dest,
    scratch,
    ds,
    t1::Int,
    field_specs,
    spans::Vector{UnitRange{Int}},
    f_label::String,
)
    G = length(spans)
    modes = [:G_separate, :one_fused]
    function runmode(m)
        if m === :G_separate
            _run_G_separate!(dest, scratch, ds, t1, field_specs, spans)
        else
            _run_one_fused!(dest, scratch, ds, t1, field_specs, spans)
        end
    end
    runmode(:G_separate)
    runmode(:one_fused)
    samples = Dict{Symbol, Vector{Float64}}(m => Float64[] for m in modes)
    for _ in 1:N_WARMUP_ROUNDS
        for m in Random.shuffle(modes)
            runmode(m)
        end
    end
    for _ in 1:N_SAMPLES
        for m in Random.shuffle(modes)
            push!(samples[m], @elapsed runmode(m))
        end
    end
    ts, ss = Statistics.median(samples[:G_separate]), samples[:G_separate]
    tf, sf = Statistics.median(samples[:one_fused]), samples[:one_fused]
    fmt(t) = string(round(t * 1000; digits=2), " ms")
    println("  G=$G  spans=$spans")
    println("    G_separate ($(G)× _into_span_list!, 1 span) median=$(fmt(ts))  min…max ms: $(round(minimum(ss)*1000;digits=2)) … $(round(maximum(ss)*1000;digits=2))")
    println("    one_fused (1×, $(G) spans)            median=$(fmt(tf))  min…max: $(round(minimum(sf)*1000;digits=2)) … $(round(maximum(sf)*1000;digits=2))")
    r = tf / ts
    println("    ratio fused/G_separate = $(round(r; digits=3))  (<1 ⇒ fused wins)")
    println("    $f_label")
end

function main()
    site = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 313
    month = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
    t1 = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    experiment = length(ARGS) >= 4 ? ARGS[4] : "amip"
    g_max = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 7

    ds = MLCD.GoogleLES.load_zarr_simulation(site, month, experiment)
    ds === nothing && error("Failed to open Zarr.")

    nx, ny, nz = _ta_nx_ny_nz(ds, t1)
    scratch = Dict{String, AbstractArray{Float32, 3}}()

    println("== benchmark_nonqc_span_groups_pipeline: site=$site month=$month t=$t1 experiment=$(repr(experiment)) ==")
    println("  nz=$nz  g_max=$g_max  z-chunk disjoint test spans (see DEFAULT_SPANS_PER_CHUNK)")
    println("  interleaved warmup=$N_WARMUP_ROUNDS timed_rounds=$N_SAMPLES")
    fe = get(ENV, "MLCD_GOOGLELES_FUSE_SPAN_Z_READS", "(default 1)")
    println("  MLCD_GOOGLELES_FUSE_SPAN_Z_READS=$(repr(fe))")

    n_fit = count(r -> last(r) <= nz, DEFAULT_SPANS_PER_CHUNK)
    max_g = min(g_max, n_fit)
    max_g < 2 && error("nz=$nz only fits $n_fit test span(s); need G>=2")

    specs1 = (("T", "ta"),)
    dest1 = _alloc_dest(specs1, nx, ny, nz)
    specsF = _non_qc_field_specs()
    destF = _alloc_dest(specsF, nx, ny, nz)

    for G in 2:max_g
        spans = _spans_for_G(nz, G)
        println("\n--- G=$G ---")
        _bench_G(dest1, scratch, ds, t1, specs1, spans, "F=1 (ta)")
        _bench_G(destF, scratch, ds, t1, specsF, spans, "F=$(length(specsF)) (all non-q_c)")
    end

    println("\nSee also: benchmark_remote_zarr_slab_reads.jl (direct _load + copyto on `ta`).")
end

main()
