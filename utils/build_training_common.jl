# Consolidated build_tabular logic for all data sources.
# This file centralizes the orchestration of data loading and processing
# into tabular Arrow format.
#
# GoogleLES Zarr: `size`/`chunks` vs `_ARRAY_DIMENSIONS` — see docs/googleles_zarr_layout.md


# utils/build_training_data.jl
# Consolidated Logic Hub for Tabular Data Generation


#=
    Here we build training data to be stored in the training data directory.

    We follow the format laid out in dataset_spec.md

    We can store data in zarr (or tabular, idk which is faster and easier yet for machine learning) format.

    We will go from 1km to native resolution.
    While ideally we could do any size reduction, for speed it should be best for us to do binary size reductions.
    So say we have a 6km x 6km domain, we can do 6km, 3km, 1.5km. The next would be 750m, which is smaller than 1km, so we stop there (but we could do this just by repeated split combines, so split all the way down to 4x4 and then combine to 2x2, then 1x1)

    The vertical is more complicated, since vertical grids may not be uniform (luckily for us it seems they (mostly) are) Let our finest resolution be 10m, and proceed to up to a 16x reduction from the LES.
    Actually i think we should go until the coarsest reduction is 400m resolution. So if we have a 6000m domain, and say dz = 10m fixed, we can do 10m, 20m, 40m, 80m, 160m, 320m. The next would be 640m, which is larger than 400m, so we stop there. Note that if the native grid is already coarser than 10m, we just start from the native grid and do binary reductions from there until we hit 400m.
    On variable/strtch grids this gets tricker so we should write the code to be robust...

    So in the example above we'd have 3 horizontal resolutions, and 6 vertical resolutions, so 18 different combinations in total.

    Note we have to do some processing to get things like tke from u, v, w, etc.

    This should be a fully parallelizeable and well documented process, that can be run on the HPC as well with slurm, can be distributed, threaded (using OhMyThreas), or just run serially. And can be stopped and resumed, restarted, etc.
    Note many levels, subcolumns etc may have no condensate. For space, we should drop them from the outputs, since those are just null data.
    Note right now, i think the cfSite provides qi and ql, but i think the LES data only provides qc. 

    Right now, GoogleLES is using    
        liquid_frac_nuc = lambda t: (t - self._t_icenuc) / (  # pylint: disable=g-long-lambda 
            self._t_freeze - self._t_icenuc)
        Which we can see by followinga chain through :
            https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/physics/thermodynamics/water.py#L943
            https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/physics/thermodynamics/water.py#L897C2-L931C1
            https://github.com/google-research/swirl-lm/blob/06aefc0f2f152c033d91a3cdbff31519afd995cd/swirl_lm/physics/thermodynamics/thermodynamics.proto#L103
        with values t_icenuc = 233.0 K, t_freeze = 273.15 K from https://github.com/CliMA/CLIMAParameters.jl/blob/main/src/Planet/planet_parameters.jl

    In principle in the future we may want to move to a CiMA type split where we use pow_icenuc as (T - T_icenuc) / (T_freeze - T_icenuc))^pow_icenuc, but for now le'ts just do the simple thing, and recreate the data how it was run.
=#

#=
    Orchestration script for processing Google LES (Swirl-LM) and cfSite high-res datasets
    into ML-ready flattened, coarse-grained tabular matrices.
=#


using .GoogleLES: GoogleLES
using .cfSites: cfSites
using .DatasetBuilder: DatasetBuilder
using .CoarseGraining: CoarseGraining
using HTTP: HTTP
using DataFrames: DataFrames
using Arrow: Arrow
using Dates: Dates

const GOOGLELES_FIELD_SPECS = (
    ("q_c", "q_c"),
    ("T", "ta"),
    ("q_t", "hus"),
    ("u", "ua"),
    ("v", "va"),
    ("w", "wa"),
    # Use the reference pressure as the canonical GoogleLES pressure feature.
    # Swirl-LM defines total pressure as p_ref + p, but the thermodynamic path
    # we mirror here already operates on p_ref, so we intentionally do not
    # reconstruct a derived total-pressure feature at export time.
    ("p_ref", "pfull"),
    ("rho", "rhoa"),
    ("theta_li", "thetali"),
)

const CF_SITES_TRANSLATION = (
    ("temperature", "ta"),
    ("qt", "hus"),
    ("u", "ua"),
    ("v", "va"),
    ("w", "wa"),
    ("p", "pfull"),
    ("rho", "rhoa"),
    ("ql", "clw"),
    ("qi", "cli"),
    ("thetali", "thetali"),
)

const GOOGLELES_FULLCASE_BYTES_LIMIT = Int(1_000_000_000)
const GOOGLELES_DEFAULT_BATCH_SIZE = 8

function _parse_bool_env(name::String, default::Bool)
    raw = get(ENV, name, default ? "1" : "0")
    return lowercase(strip(raw)) in ("1", "true", "yes", "y", "on")
end

"""
    TabularBuildOptions(; kwargs...)

Configuration for `GoogleLES.build_tabular` / `cfSites.build_tabular` and related loaders.
Use explicit keyword arguments from Julia; for shell/CLI workflows call
[`tabular_build_options_from_env`](@ref) once at the entrypoint and pass the result in.

Immutable: use [`tabular_options_with`](@ref) to copy an existing options object; for env-based setup with
overrides in one shot, use [`tabular_build_options_from_env(; kwargs...)`](@ref).
"""
Base.@kwdef struct TabularBuildOptions
    coarsening_mode::Symbol = :hybrid
    fullcase_bytes_limit::Int = GOOGLELES_FULLCASE_BYTES_LIMIT
    force_fullcase::Bool = true
    z_chunk_merge::Bool = true
    nonqc_single_fused_load::Bool = true
    fuse_span_z_reads::Bool = true
    """`auto`, `per_span`, `full`, `full_timestep`, etc. (same strings as former `MLCD_GOOGLELES_NONQC_STRATEGY`)."""
    nonqc_strategy::String = "auto"
    timestep_profile::Bool = false
    timestep_profile_each::Bool = false
    skip_finite_assert_per_chunk::Bool = false
    close_http_pools::Bool = false
    """Max valid-box outputs per axis when strides are auto-derived (default 2 = corner sampling)."""
    sliding_outputs_h::Int = 2
    sliding_outputs_v::Int = 2
    sliding_outputs_z::Int = 2
    """Subsampled horizontal scales: block `nh` count, `:sliding` window sizes, and hybrid extra-window candidate pool (see `ReductionSpecs.subsample_closed_range`). Use a value ≥ `nh_max - nh_min + 1` for a full integer ladder."""
    sliding_window_budget_h::Int = 5
end

function assert_known_nonqc_strategy(s::AbstractString)
    mode = lowercase(strip(s))
    if mode in ("full", "full_timestep", "auto", "per_span", "sparse", "minimal")
        return nothing
    end
    throw(
        ArgumentError(
            "Unknown nonqc_strategy=$(repr(s)); use one of: auto, per_span, sparse, minimal, full, full_timestep.",
        ),
    )
end

"""
    tabular_options_with(t::TabularBuildOptions; kwargs...)

Return a new [`TabularBuildOptions`](@ref) like `t` with any given fields replaced.
"""
function tabular_options_with(t::TabularBuildOptions;
    coarsening_mode = t.coarsening_mode,
    fullcase_bytes_limit = t.fullcase_bytes_limit,
    force_fullcase = t.force_fullcase,
    z_chunk_merge = t.z_chunk_merge,
    nonqc_single_fused_load = t.nonqc_single_fused_load,
    fuse_span_z_reads = t.fuse_span_z_reads,
    nonqc_strategy = t.nonqc_strategy,
    timestep_profile = t.timestep_profile,
    timestep_profile_each = t.timestep_profile_each,
    skip_finite_assert_per_chunk = t.skip_finite_assert_per_chunk,
    close_http_pools = t.close_http_pools,
    sliding_outputs_h = t.sliding_outputs_h,
    sliding_outputs_v = t.sliding_outputs_v,
    sliding_outputs_z = t.sliding_outputs_z,
    sliding_window_budget_h = t.sliding_window_budget_h,
)
    assert_known_nonqc_strategy(nonqc_strategy)
    return TabularBuildOptions(;
        coarsening_mode,
        fullcase_bytes_limit,
        force_fullcase,
        z_chunk_merge,
        nonqc_single_fused_load,
        fuse_span_z_reads,
        nonqc_strategy,
        timestep_profile,
        timestep_profile_each,
        skip_finite_assert_per_chunk,
        close_http_pools,
        sliding_outputs_h,
        sliding_outputs_v,
        sliding_outputs_z,
        sliding_window_budget_h,
    )
end

"""
    tabular_build_options_summary(opts::TabularBuildOptions) -> String

Single-line summary for logging (coarsening mode and main GoogleLES I/O knobs).
"""
function tabular_build_options_summary(o::TabularBuildOptions)::String
    return string(
        "TabularBuildOptions: coarsening_mode=", o.coarsening_mode,
        ", force_fullcase=", o.force_fullcase,
        ", z_chunk_merge=", o.z_chunk_merge,
        ", nonqc_strategy=", repr(o.nonqc_strategy),
        ", nonqc_single_fused_load=", o.nonqc_single_fused_load,
        ", fuse_span_z_reads=", o.fuse_span_z_reads,
    )
end

"""
    tabular_build_options_from_env(; kwargs...)

Read these `MLCD_*` variables once and return a [`TabularBuildOptions`](@ref) (CLI entrypoints only;
in-library code should take an explicit `TabularBuildOptions` argument). Any `kwargs` override the
corresponding env-derived fields in a **single** construction (no extra copy).

| Field | Environment variable |
|-------|----------------------|
| `coarsening_mode` | `MLCD_COARSENING_MODE` (`hybrid` default, `block`, `sliding`; unknown values **throw**) |
| `fullcase_bytes_limit` | `MLCD_GOOGLELES_FULLCASE_BYTES_LIMIT` |
| `force_fullcase` | `MLCD_GOOGLELES_FORCE_FULLCASE` |
| `z_chunk_merge` | `MLCD_GOOGLELES_Z_CHUNK_MERGE` |
| `nonqc_single_fused_load` | `MLCD_GOOGLELES_NONQC_SINGLE_FUSED_LOAD` |
| `fuse_span_z_reads` | `MLCD_GOOGLELES_FUSE_SPAN_Z_READS` |
| `nonqc_strategy` | `MLCD_GOOGLELES_NONQC_STRATEGY` |
| `timestep_profile` | `MLCD_GOOGLELES_TIMESTEP_PROFILE` |
| `timestep_profile_each` | `MLCD_GOOGLELES_TIMESTEP_PROFILE_EACH` |
| `skip_finite_assert_per_chunk` | `MLCD_SKIP_FINITE_ASSERT_PER_CHUNK` |
| `close_http_pools` | `MLCD_CLOSE_HTTP_POOLS` |
| `sliding_outputs_h` | `MLCD_SLIDING_OUTPUTS_H` |
| `sliding_outputs_v` | `MLCD_SLIDING_OUTPUTS_V` |
| `sliding_outputs_z` | `MLCD_SLIDING_OUTPUTS_Z` |
| `sliding_window_budget_h` | `MLCD_SLIDING_WINDOW_BUDGET_H` (block `nh` subsample + sliding/hybrid window budgets; set very large for full `nh_min:nh_max` blocks) |
"""
function tabular_build_options_from_env(; kw...)::TabularBuildOptions
    coarsening_raw = lowercase(strip(get(ENV, "MLCD_COARSENING_MODE", "hybrid")))
    coarsening_mode = if coarsening_raw in ("hybrid", "default")
        :hybrid
    elseif coarsening_raw in ("block", "block_truncated")
        :block
    elseif coarsening_raw == "sliding"
        :sliding
    else
        throw(
            ArgumentError(
                "MLCD_COARSENING_MODE=$(repr(coarsening_raw)) is not recognized; use hybrid, block, or sliding. " *
                "Removed modes (e.g. convolutional, binary) are no longer accepted via env.",
            ),
        )
    end
    base = (;
        coarsening_mode = coarsening_mode,
        fullcase_bytes_limit = parse(Int, get(ENV, "MLCD_GOOGLELES_FULLCASE_BYTES_LIMIT", string(GOOGLELES_FULLCASE_BYTES_LIMIT))),
        force_fullcase = _parse_bool_env("MLCD_GOOGLELES_FORCE_FULLCASE", true),
        z_chunk_merge = _parse_bool_env("MLCD_GOOGLELES_Z_CHUNK_MERGE", true),
        nonqc_single_fused_load = _parse_bool_env("MLCD_GOOGLELES_NONQC_SINGLE_FUSED_LOAD", true),
        fuse_span_z_reads = _parse_bool_env("MLCD_GOOGLELES_FUSE_SPAN_Z_READS", true),
        nonqc_strategy = String(strip(get(ENV, "MLCD_GOOGLELES_NONQC_STRATEGY", "auto"))),
        timestep_profile = _parse_bool_env("MLCD_GOOGLELES_TIMESTEP_PROFILE", false),
        timestep_profile_each = _parse_bool_env("MLCD_GOOGLELES_TIMESTEP_PROFILE_EACH", false),
        skip_finite_assert_per_chunk = _parse_bool_env("MLCD_SKIP_FINITE_ASSERT_PER_CHUNK", false),
        close_http_pools = _parse_bool_env("MLCD_CLOSE_HTTP_POOLS", false),
        sliding_outputs_h = max(1, parse(Int, get(ENV, "MLCD_SLIDING_OUTPUTS_H", "2"))),
        sliding_outputs_v = max(1, parse(Int, get(ENV, "MLCD_SLIDING_OUTPUTS_V", "2"))),
        sliding_outputs_z = max(1, parse(Int, get(ENV, "MLCD_SLIDING_OUTPUTS_Z", "2"))),
        sliding_window_budget_h = max(1, parse(Int, get(ENV, "MLCD_SLIDING_WINDOW_BUDGET_H", "5"))),
    )
    opts = TabularBuildOptions(; merge(base, (; kw...))...)
    assert_known_nonqc_strategy(opts.nonqc_strategy)
    return opts
end

function case_arrow_filename(site_id::Int, month::Int, experiment::String)
    site_id_str = lpad(string(site_id), 3, '0')
    month_str = lpad(string(month), 2, '0')
    return "googleles_case__$(site_id_str)__month__$(month_str)__exp__$(experiment).arrow"
end

function case_arrow_path(site_id::Int, month::Int, experiment::String, output_dir::String)
    return joinpath(output_dir, case_arrow_filename(site_id, month, experiment))
end

function _empty_googleles_case_df()
    return DataFrames.DataFrame(
        qt = Float32[],
        theta_li = Float32[],
        ta = Float32[],
        p = Float32[],
        rho = Float32[],
        w = Float32[],
        q_liq = Float32[],
        q_ice = Float32[],
        q_con = Float32[],
        liq_fraction = Float32[],
        ice_fraction = Float32[],
        cloud_fraction = Float32[],
        tke = Float32[],
        var_qt = Float32[],
        var_ql = Float32[],
        var_qi = Float32[],
        var_w = Float32[],
        var_h = Float32[],
        cov_qt_ql = Float32[],
        cov_qt_qi = Float32[],
        cov_qt_w = Float32[],
        cov_qt_h = Float32[],
        cov_ql_qi = Float32[],
        cov_ql_w = Float32[],
        cov_ql_h = Float32[],
        cov_qi_w = Float32[],
        cov_qi_h = Float32[],
        cov_w_h = Float32[],
        resolution_h = Float32[],
        domain_h = Float32[],
        resolution_z = Float32[],
        data_source = String[],
        month = Int[],
        cfSite_number = Int[],
        forcing_model = String[],
        experiment = String[],
    )
end

function _estimate_googleles_case_bytes(ds)
    estimated_bytes = 0
    for (g_var, _) in GOOGLELES_FIELD_SPECS
        estimated_bytes += prod(size(ds[g_var])) * sizeof(Float32)
    end
    return estimated_bytes
end

function _progress_print(prefix::AbstractString, current::Int, total::Int, label::AbstractString, started_at::Float64)
    # Pipes / file redirects are not a TTY: `\r` progress bars are invisible and can fill pipe
    # buffers if the reader exits early (e.g. `julia ... | head`). Emit throttled line logs instead.
    if !(stdout isa Base.TTY || isinteractive())
        total = max(total, 1)
        throttle = total <= 32 ? 1 : max(1, total ÷ 40)
        if current == 1 || current % throttle == 0 || current == total
            elapsed = time() - started_at
            println(
                "[progress] ",
                prefix,
                " ",
                current,
                "/",
                total,
                " ",
                label,
                " elapsed=",
                string(round(elapsed; digits=1)),
                "s",
            )
            flush(stdout)
        end
        return
    end

    width = 26
    total = max(total, 1)
    ratio = clamp(current / total, 0.0, 1.0)
    filled = min(width, max(0, round(Int, ratio * width)))
    empty = width - filled
    bar = repeat("=", max(filled - (current < total ? 1 : 0), 0)) * (current < total ? ">" : "") * repeat(" ", max(empty, 0))
    elapsed = time() - started_at
    rate = current > 0 ? elapsed / current : 0.0
    eta = current < total && rate > 0 ? (total - current) * rate : 0.0
    pct = lpad(string(round(Int, ratio * 100)), 3)
    eta_text = eta > 0 ? string(round(eta; digits=1), "s") : "0.0s"
    print(stdout, '\r', "[", bar, "] ", current, "/", total, " ", pct, "% ", prefix, " ", label, " eta=", eta_text, "   ")
    flush(stdout)
end

function _progress_finish()
    if stdout isa Base.TTY || isinteractive()
        println(stdout)
    end
end

"""
    _slice_t_range_4d(var, timestep_range, t_idx)

Slice a 4D Zarr variable along whichever Julia axis corresponds to logical time `t`.
Returns a view with the time axis retained.
"""
@inline function _slice_t_range_4d(var, timestep_range, t_idx::Int)
    if t_idx < 1 || t_idx > 4
        error("Invalid time axis index: $t_idx")
    end
    return selectdim(var, t_idx, timestep_range)
end

"""
    _perm_to_txyz(julia_dim_names)

Build the permutation that maps Julia axis order to this pipeline's internal
standard order `(t, x, y, z)`.
"""
@inline function _find_dim_idx(julia_dim_names::NTuple{4, Symbol}, target::Symbol)
    @inbounds for i in 1:4
        if julia_dim_names[i] == target
            return i
        end
    end
    return 0
end

function _perm_to_txyz(julia_dim_names::NTuple{4, Symbol})
    t_idx = _find_dim_idx(julia_dim_names, :t)
    x_idx = _find_dim_idx(julia_dim_names, :x)
    y_idx = _find_dim_idx(julia_dim_names, :y)
    z_idx = _find_dim_idx(julia_dim_names, :z)

    if t_idx == 0 || x_idx == 0 || y_idx == 0 || z_idx == 0
        error("Missing one of required dims (t,x,y,z); found dims=$(julia_dim_names)")
    end

    return (t_idx, x_idx, y_idx, z_idx)
end

"""
    _reorder_to_txyz_view(raw, julia_dim_names)

Return a lazy axis-permuted view of `raw` in this pipeline's internal `(t, x, y, z)` order.
"""
@inline function _reorder_to_txyz_view(raw, julia_dim_names::NTuple{4, Symbol})
    perm = _perm_to_txyz(julia_dim_names)
    return PermutedDimsArray(raw, perm)
end

function _load_googleles_cache(ds, timestep_range; field_specs=GOOGLELES_FIELD_SPECS)
    cache = Dict{String, AbstractArray{Float32, 4}}()
    for (g_var, c_var) in field_specs
        var = ds[g_var]
        if !haskey(var.attrs, "_ARRAY_DIMENSIONS")
            error("GoogleLES variable '$g_var' is missing _ARRAY_DIMENSIONS metadata")
        end

        metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
        if length(metadata_dim_names_raw) != 4
            error("GoogleLES variable '$g_var' expected 4 dims, found $(length(metadata_dim_names_raw))")
        end

        julia_dim_names = (
            Symbol(metadata_dim_names_raw[4]),
            Symbol(metadata_dim_names_raw[3]),
            Symbol(metadata_dim_names_raw[2]),
            Symbol(metadata_dim_names_raw[1]),
        )

        t_idx = _find_dim_idx(julia_dim_names, :t)
        if t_idx == 0
            error("GoogleLES variable '$g_var' missing required dim 't'; found dims=$(julia_dim_names)")
        end
        raw = _slice_t_range_4d(var, timestep_range, t_idx)
        canonical = _reorder_to_txyz_view(raw, julia_dim_names)

        if !(eltype(canonical) <: Float32)
            error("GoogleLES variable '$g_var' must be Float32, found eltype=$(eltype(canonical)). Refusing implicit conversion to avoid large allocations.")
        end
        cache[c_var] = canonical
    end
    return cache
end

"""
    _load_googleles_timestep_fields!(cache, ds, timestep_idx; field_specs, z_subset=nothing)

Load one GoogleLES timestep for selected fields into `cache` in canonical `(x, y, z)` order.

- `z_subset === nothing`: full `(x,y,z)` slab.
- `z_subset isa UnitRange`: contiguous z slice.
- `z_subset isa Vector{Int}`: **fused** indices along z (e.g. vcat of `10:20` and `40:50`) —
  one indexed view `[:, :, z_subset]`, **not** a widened hull. Underlying Zarr may still read
  chunks once per touched block depending on DiskArrays.

For `Float32` data, values are views until `copyto!` materializes.
"""
function _load_googleles_timestep_fields!(
    cache::Dict{String, AbstractArray{Float32, 3}},
    ds,
    timestep_idx::Int;
    field_specs=GOOGLELES_FIELD_SPECS,
    z_subset::Union{Nothing, UnitRange{Int}, Vector{Int}}=nothing,
)
    empty!(cache)
    for (g_var, c_var) in field_specs
        var = ds[g_var]
        if !haskey(var.attrs, "_ARRAY_DIMENSIONS")
            error("GoogleLES variable '$g_var' is missing _ARRAY_DIMENSIONS metadata")
        end

        metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
        if length(metadata_dim_names_raw) != 4
            error("GoogleLES variable '$g_var' expected 4 dims, found $(length(metadata_dim_names_raw))")
        end

        julia_dim_names = (
            Symbol(metadata_dim_names_raw[4]),
            Symbol(metadata_dim_names_raw[3]),
            Symbol(metadata_dim_names_raw[2]),
            Symbol(metadata_dim_names_raw[1]),
        )

        t_axis_idx = _find_dim_idx(julia_dim_names, :t)
        if t_axis_idx == 0
            error("GoogleLES variable '$g_var' missing required dim 't'; found dims=$(julia_dim_names)")
        end

        raw = _slice_t_range_4d(var, timestep_idx:timestep_idx, t_axis_idx)
        canonical = _reorder_to_txyz_view(raw, julia_dim_names)
        full_t_slice = @view canonical[1, :, :, :]
        nz = size(full_t_slice, 3)

        z_view = if isnothing(z_subset)
            full_t_slice
        elseif z_subset isa UnitRange
            (first(z_subset) >= 1 && last(z_subset) <= nz) || error("z_subset $(z_subset) out of bounds for '$g_var' (nz=$nz)")
            @view full_t_slice[:, :, z_subset]
        else
            isempty(z_subset) && error("empty z_subset for '$g_var'")
            @inbounds for k in z_subset
                (k >= 1 && k <= nz) || error("z index $k out of bounds for '$g_var' (nz=$nz)")
            end
            @view full_t_slice[:, :, z_subset]
        end

        if !(eltype(z_view) <: Float32)
            error("GoogleLES variable '$g_var' must be Float32, found eltype=$(eltype(z_view)). Refusing implicit conversion to avoid large allocations.")
        end
        cache[c_var] = z_view
    end
    return cache
end

function _load_googleles_timestep_fields(
    ds,
    timestep_idx::Int;
    field_specs=GOOGLELES_FIELD_SPECS,
    z_subset::Union{Nothing, UnitRange{Int}, Vector{Int}}=nothing,
)
    cache = Dict{String, AbstractArray{Float32, 3}}()
    return _load_googleles_timestep_fields!(cache, ds, timestep_idx; field_specs=field_specs, z_subset=z_subset)
end

"""
    _load_googleles_timestep_fields_into!(dest, ds, timestep_idx; field_specs, z_range, scratch)

Load **only** the native contiguous `z_range` from Zarr into `dest[:, :, z_range]`. No widening
to a chunk hull at this layer — each call passes the minimal slice the mask needs. Zarr may
still fetch/decode whole compressed chunks internally.
"""
function _load_googleles_timestep_fields_into!(
    dest::Dict{String, Array{Float32, 3}},
    ds,
    timestep_idx::Int;
    field_specs=GOOGLELES_FIELD_SPECS,
    z_range::UnitRange{Int},
    scratch::Dict{String, AbstractArray{Float32, 3}},
)
    for (_, c_var) in field_specs
        haskey(dest, c_var) || error("Missing destination buffer for '$c_var'")
    end
    empty!(scratch)
    _load_googleles_timestep_fields!(scratch, ds, timestep_idx; field_specs=field_specs, z_subset=z_range)
    for (_, c_var) in field_specs
        copyto!(@view(dest[c_var][:, :, z_range]), scratch[c_var])
    end
    return dest
end

"""
    _load_googleles_timestep_fields_into_span_list!(dest, ds, timestep_idx; field_specs, orig_spans, scratch)

**Chunk-merge path (default):** one overlap group → **one** `_load_googleles_timestep_fields!`
with `z_subset = vcat(collect.(orig_spans)...)` (fused vector index along z), then scatter
`copyto!` into each `dest[:, :, z_r]`. **No widened hull** (`1:60`); not two separate contiguous
slices when `length(orig_spans) > 1`.

Single-span groups use `_load_googleles_timestep_fields_into!` (same as one `UnitRange`).

Pass `fuse_span_z_reads=false` to force one `UnitRange` read per span (legacy loop).
"""
function _load_googleles_timestep_fields_into_span_list!(
    dest::Dict{String, Array{Float32, 3}},
    ds,
    timestep_idx::Int;
    field_specs=GOOGLELES_FIELD_SPECS,
    orig_spans::Vector{UnitRange{Int}},
    scratch::Dict{String, AbstractArray{Float32, 3}},
    fuse_span_z_reads::Bool=true,
)
    fuse = fuse_span_z_reads
    if length(orig_spans) == 1 || !fuse
        for z_r in orig_spans
            _load_googleles_timestep_fields_into!(
                dest,
                ds,
                timestep_idx;
                field_specs=field_specs,
                z_range=z_r,
                scratch=scratch,
            )
        end
        return dest
    end

    z_idx = Int[]
    for z_r in orig_spans
        append!(z_idx, collect(z_r))
    end
    for (_, c_var) in field_specs
        haskey(dest, c_var) || error("Missing destination buffer for '$c_var'")
    end
    empty!(scratch)
    _load_googleles_timestep_fields!(scratch, ds, timestep_idx; field_specs=field_specs, z_subset=z_idx)
    offset = 1
    for z_r in orig_spans
        L = length(z_r)
        for (_, c_var) in field_specs
            copyto!(
                @view(dest[c_var][:, :, z_r]),
                @view(scratch[c_var][:, :, offset:offset + L - 1]),
            )
        end
        offset += L
    end
    return dest
end

"""
    _materialize_googleles_nonqc_timestep_into!(dest, ds, timestep_idx; field_specs)

Read all `field_specs` for one timestep and `copyto!` the full native `(x,y,z)` slab into
`dest` (one Zarr access pattern per field). Used when `TabularBuildOptions.nonqc_strategy`
requests `full` / `full_timestep` (optional; default is per-span loads).
"""
function _materialize_googleles_nonqc_timestep_into!(
    dest::Dict{String, Array{Float32, 3}},
    ds,
    timestep_idx::Int;
    field_specs,
)
    cache = _load_googleles_cache(ds, timestep_idx:timestep_idx; field_specs=field_specs)
    for (_, c_var) in field_specs
        haskey(dest, c_var) || error("Missing destination buffer for '$c_var'")
        src_t = @view cache[c_var][1, :, :, :]
        copyto!(dest[c_var], src_t)
    end
    return dest
end

function _count_true_spans(mask::BitVector)::Int
    n = length(mask)
    k = 1
    nruns = 0
    while k <= n
        while k <= n && !mask[k]
            k += 1
        end
        k > n && break
        nruns += 1
        while k <= n && mask[k]
            k += 1
        end
    end
    return nruns
end

"""
    _googleles_use_full_nonqc_timestep_load(n_spans, n_keep, nz) -> Bool

Whether to load all non-`q_c` fields for the **entire** native column once this timestep
(`true`) or only each contiguous `z_keep_mask` span (`false`).

**Default policy (WAN / object-store first):** `auto` behaves like `per_span`. Two disjoint
cloudy slabs load two z-ranges (and only the Zarr chunks those slices touch), not
`z=1:nz`—so e.g. dry upper levels that are `false` in `z_keep_mask` are never requested.
Full-column load is **opt-in** only, for local or compute-bound runs where you accept extra
transfer.

`nonqc_strategy` (same strings as former env `MLCD_GOOGLELES_NONQC_STRATEGY`):

- `auto` (default), `per_span`, `sparse`, `minimal` → per-span loads only.
- `full`, `full_timestep` → one full-column materialization per cloudy timestep.

Any other string **throws** [`ArgumentError`](@ref) (see [`assert_known_nonqc_strategy`](@ref)).

`n_spans`, `n_keep`, `nz` are kept for call-site clarity and future policy hooks; they do not
trigger full-column loads today.
"""
function _googleles_use_full_nonqc_timestep_load(
    n_spans::Int,
    n_keep::Int,
    nz::Int;
    nonqc_strategy::AbstractString="auto",
)::Bool
    assert_known_nonqc_strategy(nonqc_strategy)
    mode = lowercase(strip(nonqc_strategy))
    return mode in ("full", "full_timestep")
end

"""
    _foreach_true_span(mask, f)

Call `f(k_start, k_end)` for each contiguous `true` run in `mask`.
"""
function _foreach_true_span(mask::BitVector, f::Function)
    n = length(mask)
    k = 1
    while k <= n
        while k <= n && !mask[k]
            k += 1
        end
        k > n && break
        k_start = k
        while k <= n && mask[k]
            k += 1
        end
        f(k_start, k - 1)
    end
    return nothing
end

@inline _foreach_true_span(f::Function, mask::BitVector) = _foreach_true_span(mask, f)

"""
    _collect_true_spans(mask::BitVector) -> Vector{UnitRange{Int}}

Contiguous runs of `true` in native z order (same as `_foreach_true_span`).
"""
function _collect_true_spans(mask::BitVector)::Vector{UnitRange{Int}}
    out = UnitRange{Int}[]
    _foreach_true_span(mask) do k_start, k_end
        push!(out, k_start:k_end)
    end
    return out
end

@inline function _z_native_span_to_storage_chunk_range(s::Int, e::Int, cz::Int)::Tuple{Int, Int}
    c_lo = div(s - 1, cz) + 1
    c_hi = div(e - 1, cz) + 1
    return (c_lo, c_hi)
end

@inline function _z_storage_chunk_hull_to_native_layer_range(c_lo::Int, c_hi::Int, cz::Int, nz::Int)::UnitRange{Int}
    z_lo = (c_lo - 1) * cz + 1
    z_hi = min(c_hi * cz, nz)
    return z_lo:z_hi
end

"""
    _googleles_storage_z_chunk_size(var) -> Int

Zarr chunk length along the storage axis named `z` (from `_ARRAY_DIMENSIONS`), for 4D fields.

**Doc:** `docs/googleles_zarr_layout.md` — metadata name order ≠ `chunks` index order; this function uses the same `julia_dim_names` mapping as the loaders.
"""
function _googleles_storage_z_chunk_size(var)::Int
    if !haskey(var.attrs, "_ARRAY_DIMENSIONS")
        error("GoogleLES variable is missing _ARRAY_DIMENSIONS (needed for z chunk size)")
    end
    metadata_dim_names_raw = var.attrs["_ARRAY_DIMENSIONS"]
    length(metadata_dim_names_raw) == 4 || error("expected 4D _ARRAY_DIMENSIONS for chunk lookup")
    chunks = getproperty(var, :metadata).chunks
    length(chunks) == 4 || error("expected 4D Zarr chunks, got length $(length(chunks))")
    # `_ARRAY_DIMENSIONS` order is NOT the same as Julia `size(var)` / `chunks` order (see
    # `_load_googleles_timestep_fields!`). Map to the same `julia_dim_names` permutation.
    julia_dim_names = (
        Symbol(metadata_dim_names_raw[4]),
        Symbol(metadata_dim_names_raw[3]),
        Symbol(metadata_dim_names_raw[2]),
        Symbol(metadata_dim_names_raw[1]),
    )
    z_julia_ax = findfirst(julia_dim_names) do sym
        Symbol(lowercase(String(sym))) === :z
    end
    cz = if z_julia_ax === nothing
        @warn "z not found in permuted dim names $julia_dim_names; using chunks[1] as z chunk."
        Int(first(chunks))
    else
        Int(chunks[z_julia_ax])
    end
    cz < 1 && error("invalid z chunk size $cz")
    return cz
end

function _googleles_effective_z_chunk_size(ds, g_var::AbstractString)::Int
    try
        return _googleles_storage_z_chunk_size(ds[g_var])
    catch
        try
            # Secondary: same layout as q_c on standard GoogleLES 4D fields.
            return _googleles_storage_z_chunk_size(ds["T"])
        catch err
            @warn "Could not read Zarr z chunk size from '$g_var' or T; using cz=1 (no chunk-based span merge)." exception=(err, catch_backtrace())
            return 1
        end
    end
end

"""
    _group_mask_spans_by_overlapping_z_chunks(spans, nz, cz) -> Vector{Vector{UnitRange{Int}}}

Partition mask spans into groups whose **storage z-chunk index intervals** overlap (transitively).
Each inner vector is the sorted list of mask spans in that group; **`build_tabular`** passes each
group to **`_load_googleles_timestep_fields_into_span_list!`** (one narrow Zarr slice per span).

`nz` and `cz` are required for chunk-index math; the hull of a group is **not** returned (tests
compute it separately if needed).
"""
function _group_mask_spans_by_overlapping_z_chunks(
    spans::Vector{UnitRange{Int}},
    nz::Int,
    cz::Int,
)::Vector{Vector{UnitRange{Int}}}
    n = length(spans)
    n == 0 && return Vector{UnitRange{Int}}[]

    c_ranges = Tuple{Int, Int}[_z_native_span_to_storage_chunk_range(first(r), last(r), cz) for r in spans]

    parent = collect(1:n)
    function ds_find(i::Int)::Int
        r = i
        while parent[r] != r
            r = parent[r]
        end
        return r
    end
    function ds_union!(a::Int, b::Int)::Nothing
        ra = ds_find(a)
        rb = ds_find(b)
        if ra != rb
            parent[rb] = ra
        end
        return nothing
    end

    @inbounds for i in 1:n
        li, hi = c_ranges[i]
        for j in (i + 1):n
            lj, hj = c_ranges[j]
            if li <= hj && lj <= hi
                ds_union!(i, j)
            end
        end
    end

    buckets = Dict{Int, Vector{Int}}()
    @inbounds for i in 1:n
        r = ds_find(i)
        push!(get!(Vector{Int}, buckets, r), i)
    end

    groups = Vector{Vector{UnitRange{Int}}}()
    for (_, idxs) in buckets
        sort!(idxs)
        orig = [spans[i] for i in idxs]
        sort!(orig; by = r -> first(r))
        push!(groups, orig)
    end
    sort!(groups; by = g -> first(first(g)))
    return groups
end

"""
    _googleles_nonqc_span_groups(spans, nz, z_cz, merge_z_chunks, single_fused_load)

Build `span_groups` for per-span non-`q_c` Zarr loads in `build_tabular`.

When `single_fused_load` is `true` (default in [`TabularBuildOptions`](@ref)),
returns `[spans]` so the timestep does **one** `_load_googleles_timestep_fields_into_span_list!`
with all mask runs fused — same z-indices as chunk-partitioning, but **no** extra passes over
`field_specs` for disjoint z-chunk groups.

When `false`, restores the previous split: chunk-overlap groups if `merge_z_chunks`, else one
group per contiguous span.
"""
function _googleles_nonqc_span_groups(
    spans::Vector{UnitRange{Int}},
    nz::Int,
    z_cz::Int,
    merge_z_chunks::Bool,
    single_fused_load::Bool,
)::Vector{Vector{UnitRange{Int}}}
    if single_fused_load && !isempty(spans)
        return [spans]
    elseif merge_z_chunks
        return _group_mask_spans_by_overlapping_z_chunks(spans, nz, z_cz)
    else
        return [[r] for r in spans]
    end
end

function _has_cloud_after_2x2(q_c::AbstractArray{<:Real, 3}; threshold::Float32=1f-10)
    nx, ny, nz = size(q_c)
    nxc = div(nx, 2)
    nyc = div(ny, 2)
    @inbounds for k in 1:nz
        for j in 1:nyc
            fj = 2j - 1
            for i in 1:nxc
                fi = 2i - 1
                mean_qc = 0.25f0 * (
                    q_c[fi, fj, k] + q_c[fi + 1, fj, k] +
                    q_c[fi, fj + 1, k] + q_c[fi + 1, fj + 1, k]
                )
                if mean_qc >= threshold
                    return true
                end
            end
        end
    end
    return false
end

function _safe_close_http_pools!(; close_http_pools::Bool=false)
    # Closing HTTP pools can race with background idle-monitor tasks in some
    # HTTP/OpenSSL versions and produce noisy "stream not initialized" errors.
    # Keep this opt-in via `TabularBuildOptions.close_http_pools`.
    close_http_pools || return

    try
        HTTP.Connections.closeall()
    catch err
        @debug "HTTP pool cleanup failed" exception=(err, catch_backtrace())
    end
end

function _assert_finite_dataframe(df::DataFrames.DataFrame, context::AbstractString)
    bad_counts = Pair{Symbol, Int}[]
    for nm in names(df)
        col = df[!, nm]
        if !(eltype(col) <: Real)
            continue
        end
        bad = count(x -> !isfinite(x), col)
        if bad > 0
            push!(bad_counts, Symbol(nm) => bad)
        end
    end

    if isempty(bad_counts)
        return
    end

    sort!(bad_counts; by=last, rev=true)
    top = bad_counts[1:min(end, 8)]
    error("Non-finite values detected in generated table ($context). Top bad columns: $(top)")
end

const GOOGLELES_BATCH_SPECS = (
    ("q_c", "q_c"),
    ("T", "ta"),
    ("q_t", "hus"),
    ("u", "ua"),
    ("v", "va"),
    ("w", "wa"),
    ("p_ref", "pfull"),
    ("rho", "rhoa"),
    ("theta_li", "thetali"),
)

function _progress_line(message::AbstractString)
    if isinteractive() || stdout isa Base.TTY
        print(stdout, "\r", message, " ")
        flush(stdout)
    else
        @info message
    end
end

function _progress_done()
    if isinteractive() || stdout isa Base.TTY
        println(stdout)
    end
end

# ------------------------------------------------------------------- #
# Google LES Implementation
# ------------------------------------------------------------------- #

