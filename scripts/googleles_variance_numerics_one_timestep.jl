#!/usr/bin/env julia
# Demo: one GoogleLES timestep of `theta_li`, spatial block coarsening matching the dataset idea.
#
# - **Pipeline-style:** `conv3d_block_mean` on `h` and on `h.*h` (Float32), then `var = mean(h²)-mean(h)²` in Float32.
# - **Shifted Float32 (fixed c):** `u = h - c`, same means on `u` and `u.*u`, then `var_u = mean(u²)-mean(u)²` (= Var(h) exactly in real arithmetic). For **stored mean(h)** you shift back: `mean(h) = mean(u) + c`. Variance needs **no** shift: `Var(h)=Var(u)`.
# - **Welford Float32:** one pass per block, textbook online variance (population / block mean form `M2/n`).
# - **Fused Float64:** one pass per block, Σh and Σh² in Float64, then `s²/n - (s/n)²` (reference).
#
# Shows spurious **negative** “variance” on the naive Float32 path vs alternatives,
# on **real** remote Zarr (needs HTTPS to GCS).
#
# Usage:
#   julia --project=. scripts/googleles_variance_numerics_one_timestep.jl
#
# Env (optional):
#   MLCD_SITE=0 MLCD_MONTH=1 MLCD_EXPERIMENT=amip MLCD_TIMESTEP=1 MLCD_FX=4 MLCD_FY=4 MLCD_FZ=2
#   MLCD_ARROW_DIR=/path/to/processed   # if set, also print `var_h < 0` rate in existing Arrow tables
#   MLCD_SHIFT=300.0                      # subtract this Float32 constant before moment accumulations (θ_li scale)
#
# This script does not modify package sources; it only `include`s `array_utils` + `GoogleLES`.

const ROOT = abspath(joinpath(@__DIR__, ".."))

include(joinpath(ROOT, "utils", "array_utils.jl"))
using .ArrayUtils: ArrayUtils

include(joinpath(ROOT, "utils", "GoogleLES.jl"))
using .GoogleLES: GoogleLES
using Zarr: Zarr
using Arrow: Arrow
using DataFrames: DataFrame

function _find_dim_idx(julia_dim_names::NTuple{4, Symbol}, target::Symbol)
    @inbounds for i in 1:4
        julia_dim_names[i] == target && return i
    end
    return 0
end

function _perm_to_txyz(julia_dim_names::NTuple{4, Symbol})
    t_idx = _find_dim_idx(julia_dim_names, :t)
    x_idx = _find_dim_idx(julia_dim_names, :x)
    y_idx = _find_dim_idx(julia_dim_names, :y)
    z_idx = _find_dim_idx(julia_dim_names, :z)
    (t_idx == 0 || x_idx == 0 || y_idx == 0 || z_idx == 0) &&
        error("Missing (t,x,y,z) in dims=$(julia_dim_names)")
    return (t_idx, x_idx, y_idx, z_idx)
end

function _slice_t_range_4d(var, timestep_range, t_idx::Int)
    return Base.selectdim(var, t_idx, timestep_range)
end

"""
Load `theta_li` for one timestep as dense `(nx,ny,nz)` Float32 in pipeline order (x fastest convention
matching `conv3d_block_mean`: indices `[i,j,k]` = x,y,z).
"""
function load_theta_li_xyz!(ds::Zarr.ZGroup, timestep_idx::Int)::Array{Float32, 3}
    haskey(ds.arrays, "theta_li") || error("Zarr group has no array \"theta_li\"")
    var = ds.arrays["theta_li"]
    haskey(var.attrs, "_ARRAY_DIMENSIONS") || error("theta_li missing _ARRAY_DIMENSIONS")
    meta = var.attrs["_ARRAY_DIMENSIONS"]
    length(meta) == 4 || error("expected 4D theta_li")
    julia_dim_names = (
        Symbol(meta[4]),
        Symbol(meta[3]),
        Symbol(meta[2]),
        Symbol(meta[1]),
    )
    t_axis_idx = _find_dim_idx(julia_dim_names, :t)
    t_axis_idx == 0 && error("no time axis in $(julia_dim_names)")
    raw = _slice_t_range_4d(var, timestep_idx:timestep_idx, t_axis_idx)
    canonical = PermutedDimsArray(raw, _perm_to_txyz(julia_dim_names))
    slab = @view canonical[1, :, :, :]
    return Array(slab)
end

"""Finite-sample variance per block: Float64 Σh, Σh² (same geometry as `conv3d_block_mean`)."""
function block_variance_f64_fused(h::Array{Float32, 3}, fx::Int, fy::Int, fz::Int)
    nx, ny, nz = size(h)
    nx % fx != 0 && error("nx=$(nx) not divisible by fx=$(fx)")
    ny % fy != 0 && error("ny=$(ny) not divisible by fy=$(fy)")
    nz % fz != 0 && error("nz=$(nz) not divisible by fz=$(fz)")
    nxo = div(nx, fx)
    nyo = div(ny, fy)
    nzo = div(nz, fz)
    out = Array{Float64}(undef, nxo, nyo, nzo)
    inv_vol = 1.0 / Float64(fx * fy * fz)
    @inbounds for k in 1:nzo
        base_k = (k - 1) * fz + 1
        for j in 1:nyo
            base_j = (j - 1) * fy + 1
            for i in 1:nxo
                base_i = (i - 1) * fx + 1
                s1 = 0.0
                s2 = 0.0
                for kk in 0:(fz - 1)
                    z = base_k + kk
                    for jj in 0:(fy - 1)
                        y = base_j + jj
                        for ii in 0:(fx - 1)
                            xi = base_i + ii
                            v = Float64(h[xi, y, z])
                            s1 += v
                            s2 += v * v
                        end
                    end
                end
                μ = s1 * inv_vol
                out[i, j, k] = s2 * inv_vol - μ * μ
            end
        end
    end
    return out
end

"""Population variance per block, single-pass Welford, all arithmetic in Float32."""
function block_variance_welford_f32(h::Array{Float32, 3}, fx::Int, fy::Int, fz::Int)
    nx, ny, nz = size(h)
    nxo = div(nx, fx)
    nyo = div(ny, fy)
    nzo = div(nz, fz)
    out = Array{Float32}(undef, nxo, nyo, nzo)
    @inbounds for k in 1:nzo
        base_k = (k - 1) * fz + 1
        for j in 1:nyo
            base_j = (j - 1) * fy + 1
            for i in 1:nxo
                base_i = (i - 1) * fx + 1
                nloc = 0f0
                meanv = 0f0
                M2 = 0f0
                for kk in 0:(fz - 1)
                    z = base_k + kk
                    for jj in 0:(fy - 1)
                        y = base_j + jj
                        for ii in 0:(fx - 1)
                            xi = base_i + ii
                            x = h[xi, y, z]
                            nloc += 1f0
                            delta = x - meanv
                            meanv += delta / nloc
                            delta2 = x - meanv
                            M2 += delta * delta2
                        end
                    end
                end
                out[i, j, k] = M2 / nloc
            end
        end
    end
    return out
end

function maxabsdiff(a::AbstractArray{<:Real}, b::AbstractArray{<:Real})
    return maximum(abs.(vec(collect(a)) .- vec(collect(b))))
end

function summarize_negatives(v::AbstractArray, name::String)
    fin = filter(isfinite, vec(collect(v)))
    isempty(fin) && return println("  $(name): no finite values")
    n = length(fin)
    nn = count(<(0), fin)
    println("  $(name): n=$(n), count(<0)=$(nn) ($(round(100 * nn / n, digits=2))%), min=$(minimum(fin)), max=$(maximum(fin))")
    return nothing
end

function maybe_summarize_arrow_dir(dir::String)
    dir = strip(dir)
    isempty(dir) && return
    isdir(dir) || (println("MLCD_ARROW_DIR is not a directory: ", dir); return)
    try
        paths = sort(filter(f -> endswith(f, ".arrow"), readdir(dir; join=true)))
        isempty(paths) && (println("No .arrow files in ", dir); return)
        # One file is enough for a symptom count
        p = first(paths)
        df = DataFrame(Arrow.Table(p))
        if !hasproperty(df, :var_h)
            println("Arrow file has no column var_h: ", p)
            return
        end
        v = Float64.(df.var_h)
        v = v[isfinite.(v)]
        n = length(v)
        nn = count(<(0), v)
        println()
        println("── Existing processed Arrow (symptom only): ", p, " ──")
        println("  var_h: n=$(n), count(<0)=$(nn) ($(n > 0 ? round(100 * nn / n, digits=2) : 0.0)%), min=$(n > 0 ? minimum(v) : NaN)")
        println("  (Arrow rows reflect the **current** builder; this demo shows Zarr-side numerics on raw θ_li.)")
        println("──")
    catch e
        println("Could not scan Arrow dir: ", sprint(showerror, e))
    end
    return nothing
end

function main()
    site = parse(Int, get(ENV, "MLCD_SITE", "0"))
    month = parse(Int, get(ENV, "MLCD_MONTH", "1"))
    experiment = get(ENV, "MLCD_EXPERIMENT", "amip")
    tidx = parse(Int, get(ENV, "MLCD_TIMESTEP", "1"))
    fx = parse(Int, get(ENV, "MLCD_FX", "4"))
    fy = parse(Int, get(ENV, "MLCD_FY", "4"))
    fz = parse(Int, get(ENV, "MLCD_FZ", "2"))
    c = parse(Float32, get(ENV, "MLCD_SHIFT", "300.0"))

    println("Opening GoogleLES Zarr (network): site=$(site) month=$(month) experiment=$(experiment)")
    ds = GoogleLES.load_zarr_simulation(site, month, experiment)
    ds === nothing && error("load_zarr_simulation returned nothing (network or path failure)")

    println("Loading theta_li timestep ", tidx, " …")
    h = load_theta_li_xyz!(ds, tidx)
    println("  Array size (x,y,z) = ", size(h), ", eltype = ", eltype(h))

    println("Block factors (fx,fy,fz) = (", fx, ",", fy, ",", fz, ")")
    mean_h = ArrayUtils.conv3d_block_mean(h, fx, fy, fz)
    h2 = similar(h)
    @inbounds for i in eachindex(h)
        h2[i] = h[i] * h[i]
    end
    mean_h2 = ArrayUtils.conv3d_block_mean(h2, fx, fy, fz)
    var_f32 = mean_h2 .- mean_h .* mean_h

    var_f64 = block_variance_f64_fused(h, fx, fy, fz)

    # Fixed-offset shift: same moment pipeline on u = h - c (all Float32).
    u = similar(h)
    @inbounds for i in eachindex(h)
        u[i] = h[i] - c
    end
    mean_u = ArrayUtils.conv3d_block_mean(u, fx, fy, fz)
    u2 = similar(u)
    @inbounds for i in eachindex(u)
        u2[i] = u[i] * u[i]
    end
    mean_u2 = ArrayUtils.conv3d_block_mean(u2, fx, fy, fz)
    var_f32_shifted = mean_u2 .- mean_u .* mean_u
    mean_h_from_shift = mean_u .+ c

    var_welford_f32 = block_variance_welford_f32(h, fx, fy, fz)

    println()
    println("── Comparison on this timestep (same blocks); shift c = ", c, " ──")
    summarize_negatives(var_f32, "Pipeline Float32  ⟨h²⟩−⟨h⟩² (naive)")
    summarize_negatives(var_f32_shifted, "Shifted Float32  u=h−c, ⟨u²⟩−⟨u⟩²  (Var(h)=Var(u))")
    summarize_negatives(var_welford_f32, "Welford Float32  per-block M2/n")
    summarize_negatives(var_f64, "Fused Float64 (reference)")
    println("  max|mean_h − (⟨u⟩+c)| : ", maxabsdiff(mean_h, mean_h_from_shift), "  (shift-back check for stored means)")
    println("  max|var_shifted − var_f64| : ", maxabsdiff(var_f32_shifted, var_f64))
    println("  max|var_welford − var_f64| : ", maxabsdiff(var_welford_f32, var_f64))
    println("  Note: for Arrow/schema, store mean(h)=⟨u⟩+c; store var(h)=⟨u²⟩−⟨u⟩² (do not add c to variance).")
    println("── end ──")

    arrow_dir = get(ENV, "MLCD_ARROW_DIR", "")
    maybe_summarize_arrow_dir(arrow_dir)
    return nothing
end

main()
