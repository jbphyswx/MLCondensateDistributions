"""
Serial throughput probe for `DatasetBuilder.process_abstract_chunk` (synthetic LES-sized chunk).

Usage:
    julia --project=<repo_root> \\
      <repo_root>/experiments/amip_baseline/profile/profile_dataset_builder_breakdown.jl

Env:
    NX, NY, NZ   — grid shape (defaults 124, 124, 120)
    DX           — native Δx [m] (default 50)
    MIN_H        — min horizontal resolution [m] (default 1000)
    REPEATS      — timed iterations after one warmup (default 3)
    PROFILE=1    — run `Profile.@profile` + `Profile.print` on last repeat
    SEED         — RNG seed (default 42)
"""

using Random: Random
using Profile: Profile
using Statistics: Statistics

const ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "src", "MLCondensateDistributions.jl"))
using .MLCondensateDistributions: MLCondensateDistributions as MLCD

const NX = parse(Int, get(ENV, "NX", "124"))
const NY = parse(Int, get(ENV, "NY", "124"))
const NZ = parse(Int, get(ENV, "NZ", "120"))
const DX = parse(Float32, get(ENV, "DX", "50"))
const MIN_H = parse(Float32, get(ENV, "MIN_H", "1000"))
const REPEATS = parse(Int, get(ENV, "REPEATS", "3"))
const SEED = parse(Int, get(ENV, "SEED", "42"))
const DO_PROFILE = parse(Bool, get(ENV, "PROFILE", "0"))

function synthetic_fine_fields(rng::Random.AbstractRNG)
    fine_fields = Dict{String, Array{Float32, 3}}()
    for key in ("hus", "thetali", "ta", "pfull", "rhoa", "ua", "va", "wa", "clw", "cli")
        fine_fields[key] = Random.rand(rng, Float32, NX, NY, NZ)
    end
    # Ensure some "cloud" signal so the builder does real work
    fine_fields["clw"] .*= 1f-6
    fine_fields["cli"] .*= 1f-7
    fine_fields["clw"][1:2:end, 1:2:end, NZ ÷ 4:3NZ ÷ 4] .+= 5f-6
    return fine_fields
end

function main()
    rng = Random.MersenneTwister(SEED)
    fine_fields = synthetic_fine_fields(rng)
    dz = fill(10f0, NZ)
    dz[end] = 10f0
    spatial_info = (;
        dx_native = DX,
        domain_h = DX * Float32(NX - 1),
        min_h_resolution = MIN_H,
        dz_native_profile = dz,
        seeds_h = (1,),
    )
    metadata = (;
        data_source = "profile",
        month = 1,
        cfSite_number = 0,
        forcing_model = "none",
        experiment = "profile",
        verbose = false,
    )

    # Warmup (JIT)
    MLCD.DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)

    times = Float64[]
    for r in 1:REPEATS
        t = @elapsed MLCD.DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
        push!(times, t)
        if DO_PROFILE && r == REPEATS
            Profile.clear()
            Profile.@profile MLCD.DatasetBuilder.process_abstract_chunk(fine_fields, metadata, spatial_info)
        end
    end

    println("process_abstract_chunk: NX=$NX NY=$NY NZ=$NZ DX=$DX MIN_H=$MIN_H")
    println("  repeats=$(REPEATS) seconds: ", join(round.(times; digits=3), ", "))
    println("  median: ", round(Statistics.median(times); digits=3), " s")
    if DO_PROFILE
        println("\n--- Profile.print (last repeat + profile pass) ---")
        Profile.print(maxdepth = 30)
    end
end

main()
