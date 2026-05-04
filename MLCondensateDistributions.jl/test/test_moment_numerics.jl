using Test: Test
using Random: Random
using Statistics: Statistics

include(joinpath(@__DIR__, "..", "utils", "statistical_methods", "StatisticalMethods.jl"))
using .StatisticalMethods: StatisticalMethods

if !isdefined(Main, :ArrayUtils)
    include(joinpath(@__DIR__, "..", "utils", "array_utils.jl"))
end
using .ArrayUtils: ArrayUtils

include(joinpath(@__DIR__, "..", "utils", "coarsening_pipeline.jl"))
using .CoarseningPipeline: CoarseningPipeline

Test.@testset "StatisticalMethods Chan variance" begin
    rng = Random.MersenneTwister(2026)
    for _ in 1:20
        x1 = randn(rng, 15)
        x2 = randn(rng, 23)
        n1, n2 = length(x1), length(x2)
        μ1 = sum(x1) / n1
        μ2 = sum(x2) / n2
        M2_1 = sum((xi - μ1)^2 for xi in x1)
        M2_2 = sum((xi - μ2)^2 for xi in x2)
        n, μ, M2 = StatisticalMethods.merge_variance_chan(n1, μ1, M2_1, n2, μ2, M2_2)
        x = vcat(x1, x2)
        μ_ref = sum(x) / length(x)
        M2_ref = sum((xi - μ_ref)^2 for xi in x)
        Test.@test n == n1 + n2
        Test.@test μ ≈ μ_ref rtol = 1e-12
        Test.@test M2 ≈ M2_ref rtol = 1e-10
    end
end

Test.@testset "StatisticalMethods Pebay covariance" begin
    rng = Random.MersenneTwister(404)
    for _ in 1:15
        x1 = randn(rng, 12)
        y1 = randn(rng, 12)
        x2 = randn(rng, 18)
        y2 = randn(rng, 18)
        n1, n2 = length(x1), length(x2)
        μx1, μy1 = sum(x1) / n1, sum(y1) / n1
        μx2, μy2 = sum(x2) / n2, sum(y2) / n2
        C1 = sum((x1[i] - μx1) * (y1[i] - μy1) for i in 1:n1)
        C2 = sum((x2[i] - μx2) * (y2[i] - μy2) for i in 1:n2)
        n, μx, μy, C = StatisticalMethods.merge_covariance_pebay(n1, μx1, μy1, C1, n2, μx2, μy2, C2)
        x = vcat(x1, x2)
        y = vcat(y1, y2)
        μxr = sum(x) / length(x)
        μyr = sum(y) / length(y)
        Cr = sum((x[i] - μxr) * (y[i] - μyr) for i in eachindex(x))
        Test.@test n == n1 + n2
        Test.@test μx ≈ μxr rtol = 1e-12
        Test.@test μy ≈ μyr rtol = 1e-12
        Test.@test C ≈ Cr rtol = 1e-9
    end
end

Test.@testset "ArrayUtils conv3d_block_mean_M2 vs population variance" begin
    rng = Random.MersenneTwister(99)
    data = randn(rng, Float32, 8, 8, 4)
    μ, M2 = ArrayUtils.conv3d_block_mean_M2(data, 2, 2, 2)
    n = 2 * 2 * 2
    @inbounds for k in 1:size(μ, 3)
        for j in 1:size(μ, 2)
            for i in 1:size(μ, 1)
                bi = 2(i - 1) + 1
                bj = 2(j - 1) + 1
                bk = 2(k - 1) + 1
                blk = vec(data[bi:bi+1, bj:bj+1, bk:bk+1])
                v_ref = Statistics.var(blk; corrected=false)
                Test.@test M2[i, j, k] / n ≈ v_ref rtol = 1e-5 atol = 1e-6
            end
        end
    end
end

Test.@testset "CoarseningPipeline horizontal native M2 matches one-shot 3D M2" begin
    rng = Random.MersenneTwister(7)
    nh = 4
    nx, ny, nz = 32, 32, 8
    qt = randn(rng, Float32, nx, ny, nz)
    fields = (
        hus = qt,
        thetali = randn(rng, Float32, nx, ny, nz),
        ta = randn(rng, Float32, nx, ny, nz),
        pfull = randn(rng, Float32, nx, ny, nz),
        rhoa = randn(rng, Float32, nx, ny, nz),
        ua = randn(rng, Float32, nx, ny, nz),
        va = randn(rng, Float32, nx, ny, nz),
        wa = randn(rng, Float32, nx, ny, nz),
        clw = randn(rng, Float32, nx, ny, nz),
        cli = randn(rng, Float32, nx, ny, nz),
        liq_fraction = rand(rng, Float32, nx, ny, nz),
        ice_fraction = rand(rng, Float32, nx, ny, nz),
        cloud_fraction = rand(rng, Float32, nx, ny, nz),
    )
    pairs = (
        qt_qt = (:hus, :hus),
        ql_ql = (:clw, :clw),
        qi_qi = (:cli, :cli),
        u_u = (:ua, :ua),
        v_v = (:va, :va),
        w_w = (:wa, :wa),
        h_h = (:thetali, :thetali),
        qt_ql = (:hus, :clw),
        qt_qi = (:hus, :cli),
        qt_w = (:hus, :wa),
        qt_h = (:hus, :thetali),
        ql_qi = (:clw, :cli),
        ql_w = (:clw, :wa),
        ql_h = (:clw, :thetali),
        qi_w = (:cli, :wa),
        qi_h = (:cli, :thetali),
        w_h = (:wa, :thetali),
    )
    mom_h = CoarseningPipeline.coarsen_products_moments_horizontal_native(fields, pairs, nh)
    _, M2_3d = ArrayUtils.conv3d_block_mean_M2(qt, nh, nh, 1)
    Test.@test mom_h.qt_qt ≈ M2_3d
    means_h = CoarseningPipeline.coarsen_fields_at_level(fields, nh, nh)
    mom_chain = CoarseningPipeline.coarsen_moments_horizontal_merge(
        means_h,
        mom_h,
        pairs,
        2,
        2,
        nh * nh,
    )
    _, M2_direct = ArrayUtils.conv3d_block_mean_M2(qt, nh * 2, nh * 2, 1)
    Test.@test mom_chain.qt_qt ≈ M2_direct rtol = 1e-4 atol = 1e-5
end
