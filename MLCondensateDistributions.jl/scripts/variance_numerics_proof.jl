#!/usr/bin/env julia
# Standalone numerics demo — **not** part of the package test suite (`test/runtests.jl`).
#
# Motivation: same issue as coarse `var_h = E[h²]-E[h]²` when block means use Float32 sums.
# This file does not import MLCD; it only shows the phenomenon on synthetic data.
#
# Run:
#   julia --project=. scripts/variance_numerics_proof.jl
#
# Quiet output (checks only):
#   MLCD_QUIET_VARIANCE_NUMERICS=1 julia --project=. scripts/variance_numerics_proof.jl

using Random: Random
using Printf: @sprintf

function f32_block_mean_sum_sq_and_var(h::AbstractVector{Float32})
    n = length(h)
    invn = 1f0 / Float32(n)
    acc1 = 0f0
    acc2 = 0f0
    @inbounds for i in 1:n
        hi = h[i]
        acc1 += hi
        acc2 += hi * hi
    end
    mh = acc1 * invn
    mhh = acc2 * invn
    return mh, mhh, mhh - mh * mh
end

function f64_sample_variance(h::AbstractVector{Float32})
    n = length(h)
    s1 = 0.0
    s2 = 0.0
    @inbounds for i in 1:n
        x = Float64(h[i])
        s1 += x
        s2 += x * x
    end
    return s2 / n - (s1 / n)^2
end

function f64_block_mean_sum_sq_and_var(h::AbstractVector{Float32})
    n = length(h)
    s1 = 0.0
    s2 = 0.0
    @inbounds for i in 1:n
        x = Float64(h[i])
        s1 += x
        s2 += x * x
    end
    mh = s1 / n
    mhh = s2 / n
    return mh, mhh, mhh - mh * mh
end

function print_proof(h, v_ref, mh_f32, mhh_f32, v_f32, v_f64_acc, v_sub_f64)
    println()
    println("── Variance numerics proof (same ", length(h), " Float32 samples: h = 300 + 0.002·N(0,1)) ──")
    println("  Reference sample variance (Float64 Σh, Σh², then s² = E[h²]-E[h]²):  ", @sprintf("%.6e", v_ref))
    println("  Float32 accumulators + Float32 mean(h), mean(h²), then var in Float32: ", v_f32)
    println("  → wrong sign and ~0.1 error while truth is ~1e-6 (catastrophic cancellation vs huge ~9e4 terms).")
    println("  Same formula but Float64 accumulators for Σh, Σh²:            ", @sprintf("%.6e", v_f64_acc))
    println("  Float64 subtraction only, using Float32-rounded means:       ", v_sub_f64)
    println("  (mean_h, mean_h²) stored as Float32:  ", mh_f32, " , ", mhh_f32)
    println("  Conclusion: precision must enter **before** the coarse means are rounded (or use a fused statistic).")
    println("── end proof ──")
    println()
    return nothing
end

function main()
    Random.seed!(7)
    h = vec(300f0 .+ 0.002f0 .* randn(Float32, 64, 64))

    v_ref = f64_sample_variance(h)
    mh_f32, mhh_f32, v_f32 = f32_block_mean_sum_sq_and_var(h)
    _, _, v_f64_acc = f64_block_mean_sum_sq_and_var(h)
    v_sub_f64 = Float64(mhh_f32) - Float64(mh_f32)^2

    if get(ENV, "MLCD_QUIET_VARIANCE_NUMERICS", "") != "1"
        print_proof(h, v_ref, mh_f32, mhh_f32, v_f32, v_f64_acc, v_sub_f64)
    end

    @assert v_ref > 0
    @assert v_ref < 1f-2
    @assert v_f32 < 0
    @assert abs(v_f32) > 100 * v_ref
    @assert v_f64_acc >= -1e-10
    @assert isapprox(v_f64_acc, v_ref; rtol=0, atol=1e-9)
    @assert v_sub_f64 < 0
    @assert v_sub_f64 == Float64(v_f32)

    get(ENV, "MLCD_QUIET_VARIANCE_NUMERICS", "") == "1" || println("All checks passed.")
    return nothing
end

main()
