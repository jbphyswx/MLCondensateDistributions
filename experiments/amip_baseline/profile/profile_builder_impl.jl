"""
Profile old vs implementation dataset builder on a synthetic chunk with reproducible stage breakdown.

Usage (recommended):
    julia --project=/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions \
      /home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/profile/profile_v2_builder.jl

Optional env vars:
    NX=124
    NY=124
    NZ=120
    DX=50
    MIN_H=1000
    REPEATS=3
    SEED=42
"""

using Random
using DataFrames

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions

const NX = parse(Int, get(ENV, "NX", "124"))
const NY = parse(Int, get(ENV, "NY", "124"))
const NZ = parse(Int, get(ENV, "NZ", "120"))
const DX = parse(Float32, get(ENV, "DX", "50"))
const MIN_H = parse(Float32, get(ENV, "MIN_H", "1000"))
const REPEATS = parse(Int, get(ENV, "REPEATS", "3"))
const SEED = parse(Int, get(ENV, "SEED", "42"))

const DB = MLCondensateDistributions.DatasetBuilder
const DBI = MLCondensateDistributions.DatasetBuilderImpl
const CPI = MLCondensateDistributions.DatasetBuilderImpl.CoarseningPipeline

function make_inputs()
    Random.seed!(SEED)

    fine_fields = Dict{String, Array{Float32, 3}}(
        "hus" => rand(Float32, NX, NY, NZ) .* 1f-2,
        "thetali" => rand(Float32, NX, NY, NZ) .* 300f0,
        "ta" => rand(Float32, NX, NY, NZ) .* 300f0,
        "pfull" => rand(Float32, NX, NY, NZ) .* 1f5,
        "rhoa" => rand(Float32, NX, NY, NZ),
        "ua" => randn(Float32, NX, NY, NZ),
        "va" => randn(Float32, NX, NY, NZ),
        "wa" => randn(Float32, NX, NY, NZ),
        "clw" => rand(Float32, NX, NY, NZ) .* 1f-5,
        "cli" => rand(Float32, NX, NY, NZ) .* 1f-5,
    )

    metadata = Dict{Symbol, Any}(
        :data_source => "bench",
        :month => 1,
        :cfSite_number => 1,
        :forcing_model => "bench",
        :experiment => "bench",
        :verbose => false,
    )

    spatial_info = Dict{Symbol, Any}(
        :dx_native => DX,
        :domain_h => Float32((NX - 1) * DX),
        :min_h_resolution => MIN_H,
        :dz_native_profile => fill(Float32(20), NZ),
    )

    return fine_fields, metadata, spatial_info
end

function build_impl_stage_inputs(fine_fields, spatial_info)
    ct = DB.CLOUD_PRESENCE_THRESHOLD

    base_qt = DBI._as_f32_array3(fine_fields["hus"])
    base_h = DBI._as_f32_array3(fine_fields["thetali"])
    base_ta = DBI._as_f32_array3(fine_fields["ta"])
    base_p = DBI._as_f32_array3(fine_fields["pfull"])
    base_rho = DBI._as_f32_array3(fine_fields["rhoa"])
    base_u = DBI._as_f32_array3(fine_fields["ua"])
    base_v = DBI._as_f32_array3(fine_fields["va"])
    base_w = DBI._as_f32_array3(fine_fields["wa"])
    base_ql = DBI._as_f32_array3(fine_fields["clw"])
    base_qi = DBI._as_f32_array3(fine_fields["cli"])

    base_liq_presence = similar(base_ql)
    base_ice_presence = similar(base_qi)
    base_cloud_presence = similar(base_ql)
    @inbounds for idx in eachindex(base_ql)
        liq_01 = DB._indicator_01(base_ql[idx], ct)
        ice_01 = DB._indicator_01(base_qi[idx], ct)
        base_liq_presence[idx] = liq_01
        base_ice_presence[idx] = ice_01
        base_cloud_presence[idx] = (liq_01 > 0f0 || ice_01 > 0f0) ? 1f0 : 0f0
    end

    fields = Dict{String, Array{Float32, 3}}(
        "hus" => base_qt,
        "thetali" => base_h,
        "ta" => base_ta,
        "pfull" => base_p,
        "rhoa" => base_rho,
        "ua" => base_u,
        "va" => base_v,
        "wa" => base_w,
        "clw" => base_ql,
        "cli" => base_qi,
        "liq_fraction" => base_liq_presence,
        "ice_fraction" => base_ice_presence,
        "cloud_fraction" => base_cloud_presence,
    )

    product_pairs = Dict{String, Tuple{String, String}}(
        "qt_qt" => ("hus", "hus"),
        "ql_ql" => ("clw", "clw"),
        "qi_qi" => ("cli", "cli"),
        "u_u" => ("ua", "ua"),
        "v_v" => ("va", "va"),
        "w_w" => ("wa", "wa"),
        "h_h" => ("thetali", "thetali"),
        "qt_ql" => ("hus", "clw"),
        "qt_qi" => ("hus", "cli"),
        "qt_w" => ("hus", "wa"),
        "qt_h" => ("hus", "thetali"),
        "ql_qi" => ("clw", "cli"),
        "ql_w" => ("clw", "wa"),
        "ql_h" => ("clw", "thetali"),
        "qi_w" => ("cli", "wa"),
        "qi_h" => ("cli", "thetali"),
        "w_h" => ("wa", "thetali"),
    )

    return fields, product_pairs
end

function run_benchmark()
    fine_fields, metadata, spatial_info = make_inputs()
    fields, product_pairs = build_impl_stage_inputs(fine_fields, spatial_info)

    oldf = DB.process_abstract_chunk
    newf = DBI.process_abstract_chunk_impl

    # Warmup
    oldf(fine_fields, metadata, spatial_info)
    newf(fine_fields, metadata, spatial_info)

    println("== Old vs Impl Timing ==")
    for i in 1:REPEATS
        told = @elapsed dfo = oldf(fine_fields, metadata, spatial_info)
        tnew = @elapsed dfn = newf(fine_fields, metadata, spatial_info)
        println("run=", i, " old_s=", told, " new_s=", tnew, " old_rows=", nrow(dfo), " new_rows=", nrow(dfn))
    end

    aold = @allocated oldf(fine_fields, metadata, spatial_info)
    anew = @allocated newf(fine_fields, metadata, spatial_info)

    hv = CPI.build_horizontal_multilevel_views(
        fields,
        Float32(spatial_info[:dx_native]);
        seeds=(1,),
        min_h=Float32(spatial_info[:min_h_resolution]),
        include_full_domain=false,
        product_pairs=product_pairs,
    )

    ahviews = @allocated CPI.build_horizontal_multilevel_views(
        fields,
        Float32(spatial_info[:dx_native]);
        seeds=(1,),
        min_h=Float32(spatial_info[:min_h_resolution]),
        include_full_domain=false,
        product_pairs=product_pairs,
    )

    # One-level stage cost (representative, not full builder)
    h1 = hv[1]
    v_qt = h1.means["hus"]
    v_h = h1.means["thetali"]
    v_ta = h1.means["ta"]
    v_p = h1.means["pfull"]
    v_rho = h1.means["rhoa"]
    v_w = h1.means["wa"]
    v_ql = h1.means["clw"]
    v_qi = h1.means["cli"]
    v_liq = h1.means["liq_fraction"]
    v_ice = h1.means["ice_fraction"]
    v_cloud = h1.means["cloud_fraction"]

    vp_uu = h1.products["u_u"]
    vp_vv = h1.products["v_v"]
    vp_ww = h1.products["w_w"]
    vp_hh = h1.products["h_h"]
    vp_qtqt = h1.products["qt_qt"]
    vp_qlql = h1.products["ql_ql"]
    vp_qiqi = h1.products["qi_qi"]
    vp_qtql = h1.products["qt_ql"]
    vp_qtqi = h1.products["qt_qi"]
    vp_qtw = h1.products["qt_w"]
    vp_qth = h1.products["qt_h"]
    vp_qlqi = h1.products["ql_qi"]
    vp_qlw = h1.products["ql_w"]
    vp_qlh = h1.products["ql_h"]
    vp_qiw = h1.products["qi_w"]
    vp_qih = h1.products["qi_h"]
    vp_wh = h1.products["w_h"]

    amoments = @allocated begin
        tke = similar(v_qt)
        DBI._tke_from_moments!(tke, vp_uu, h1.means["ua"], vp_vv, h1.means["va"], vp_ww, v_w)

        var_qt = similar(v_qt)
        var_ql = similar(v_qt)
        var_qi = similar(v_qt)
        var_w = similar(v_qt)
        var_h = similar(v_qt)
        DBI._covariance_from_moments!(var_qt, vp_qtqt, v_qt, v_qt)
        DBI._covariance_from_moments!(var_ql, vp_qlql, v_ql, v_ql)
        DBI._covariance_from_moments!(var_qi, vp_qiqi, v_qi, v_qi)
        DBI._covariance_from_moments!(var_w, vp_ww, v_w, v_w)
        DBI._covariance_from_moments!(var_h, vp_hh, v_h, v_h)

        cov_qt_ql = similar(v_qt)
        cov_qt_qi = similar(v_qt)
        cov_qt_w = similar(v_qt)
        cov_qt_h = similar(v_qt)
        cov_ql_qi = similar(v_qt)
        cov_ql_w = similar(v_qt)
        cov_ql_h = similar(v_qt)
        cov_qi_w = similar(v_qt)
        cov_qi_h = similar(v_qt)
        cov_w_h = similar(v_qt)
        DBI._covariance_from_moments!(cov_qt_ql, vp_qtql, v_qt, v_ql)
        DBI._covariance_from_moments!(cov_qt_qi, vp_qtqi, v_qt, v_qi)
        DBI._covariance_from_moments!(cov_qt_w, vp_qtw, v_qt, v_w)
        DBI._covariance_from_moments!(cov_qt_h, vp_qth, v_qt, v_h)
        DBI._covariance_from_moments!(cov_ql_qi, vp_qlqi, v_ql, v_qi)
        DBI._covariance_from_moments!(cov_ql_w, vp_qlw, v_ql, v_w)
        DBI._covariance_from_moments!(cov_ql_h, vp_qlh, v_ql, v_h)
        DBI._covariance_from_moments!(cov_qi_w, vp_qiw, v_qi, v_w)
        DBI._covariance_from_moments!(cov_qi_h, vp_qih, v_qi, v_h)
        DBI._covariance_from_moments!(cov_w_h, vp_wh, v_w, v_h)
    end

    amasks_flat = @allocated begin
        ct = DB.CLOUD_PRESENCE_THRESHOLD
        q_con = v_ql .+ v_qi
        z_schemes = MLCondensateDistributions.CoarseGraining.compute_z_coarsening_scheme(spatial_info[:dz_native_profile], 400f0)
        empty_z_levels = MLCondensateDistributions.CoarseGraining.identify_empty_z_levels(q_con, ct)
        future = Int[z_schemes[i][1] for i in 2:length(z_schemes)]
        z_keep = MLCondensateDistributions.CoarseGraining.build_z_level_keep_mask(empty_z_levels, z_schemes[1][1], future)

        sparse = BitArray(undef, size(v_ql))
        @inbounds for idx in eachindex(v_ql)
            sparse[idx] = (v_ql[idx] + v_qi[idx]) < ct
        end
        zdrop = falses(size(v_ql))
        @inbounds for k in eachindex(z_keep)
            if !z_keep[k]
                zdrop[:, :, k] .= true
            end
        end

        nonfinite = falses(size(v_ql))
        combined = BitArray(undef, size(v_ql))
        @inbounds for idx in eachindex(v_ql)
            combined[idx] = sparse[idx] || nonfinite[idx] || zdrop[idx]
        end

        df = DataFrame()
        DB.flatten_and_filter!(
            df,
            combined,
            v_qt,
            v_h,
            v_ta,
            v_p,
            v_rho,
            v_w,
            v_ql,
            v_qi,
            v_liq,
            v_ice,
            v_cloud,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            v_qt,
            spatial_info[:dz_native_profile],
            Float32(1000),
            Float32(spatial_info[:domain_h]),
            metadata,
        )
    end

    println("== Allocation Breakdown (bytes) ==")
    println("hviews=", ahviews, " moments=", amoments, " masks_flat=", amasks_flat)
    println("alloc_total_old=", aold, " alloc_total_new=", anew)
end

println("== profile_v2_builder ==")
println("NX=", NX, " NY=", NY, " NZ=", NZ, " DX=", DX, " MIN_H=", MIN_H, " REPEATS=", REPEATS, " SEED=", SEED)
run_benchmark()
