using Test: Test

include("../utils/dynamics.jl")
using .Dynamics: Dynamics
using Statistics: Statistics

Test.@testset "Dynamics TKE helpers" begin
    Test.@test Dynamics.KE(3.0f0, 4.0f0, 0.0f0) == 12.5f0
    Test.@test Dynamics.variance_from_moments(5.0f0, 2.0f0) == 1.0f0
    Test.@test Dynamics.TKE(1.0f0, 2.0f0, 3.0f0) == 3.0f0
    Test.@test Dynamics.TKE_from_moments(5.0f0, 2.0f0, 6.0f0, 1.0f0, 7.0f0, 1.0f0) == 6.0f0

    mean_sq_u = Float32[1.0, 2.0, 3.0]
    mean_u = Float32[0.0, 1.0, 1.0]
    mean_sq_v = Float32[0.5, 1.5, 2.5]
    mean_v = Float32[0.0, 0.0, 0.5]
    mean_sq_w = Float32[0.25, 0.75, 1.25]
    mean_w = Float32[0.0, 0.5, 0.5]

    tke = Dynamics.TKE_from_moments(mean_sq_u, mean_u, mean_sq_v, mean_v, mean_sq_w, mean_w)
    expected = 0.5f0 .* (
        (mean_sq_u .- mean_u .* mean_u) .+
        (mean_sq_v .- mean_v .* mean_v) .+
        (mean_sq_w .- mean_w .* mean_w)
    )

    Test.@test tke == expected
    Test.@test all(tke .>= 0f0)

    # Regression: signed means can cancel, but TKE from fluctuations must remain nonnegative.
    u_raw = Float32[-10, -8]
    v_raw = Float32[1, 3]
    w_raw = Float32[0, 2]

    tke_raw_default = Dynamics.TKE(u_raw, v_raw, w_raw)
    tke_raw_explicit = Dynamics.TKE(u_raw, v_raw, w_raw, Statistics.mean(u_raw), Statistics.mean(v_raw), Statistics.mean(w_raw))
    naive_signed_sum = 0.5f0 * (sum(u_raw) + sum(v_raw) + sum(w_raw))

    Test.@test tke_raw_default == tke_raw_explicit
    Test.@test tke_raw_default > 0f0
    Test.@test naive_signed_sum < 0f0
end
