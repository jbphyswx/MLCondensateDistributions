"""
Cold-process q_c z-window scaling benchmark.

For each (z_layers, repeat), this script launches a fresh Julia subprocess,
performs one untimed warmup read, then times one read. This reduces same-process
cache artifacts and JIT-first-touch bias in the reported sample.

Usage:
    julia --project=. experiments/amip_baseline/profile/profile_qc_z_window_scaling_cold.jl

Optional env vars:
    SITE_ID=320
    MONTH=1
    EXPERIMENT=amip
    Z_LIST=1,2,4,8,16,32,64,128,256,320,480
    REPEATS=3
"""

const SITE_ID = parse(Int, get(ENV, "SITE_ID", "320"))
const MONTH = parse(Int, get(ENV, "MONTH", "1"))
const EXPERIMENT = get(ENV, "EXPERIMENT", "amip")
const REPEATS = parse(Int, get(ENV, "REPEATS", "3"))

function parse_int_list(raw::String)
    vals = Int[]
    for token in split(raw, ',')
        s = strip(token)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    sort!(unique!(vals))
    return vals
end

function median_val(values::Vector{Float64})
    s = sort(values)
    n = length(s)
    if isodd(n)
        return s[(n + 1) >>> 1]
    end
    mid = n >>> 1
    return 0.5 * (s[mid] + s[mid + 1])
end

function run_point(z_layers::Int, t_idx::Int)
    code = """
include(\"src/MLCondensateDistributions.jl\")
using .MLCondensateDistributions

ds = MLCondensateDistributions.GoogleLES.load_zarr_simulation($SITE_ID, $MONTH, \"$EXPERIMENT\")
q = ds[\"q_c\"]
nz, nx, ny, nt = size(q)
ztop = min($z_layers, nz)
t = min(max($t_idx, 1), nt)

_ = Array(@view q[1:ztop, :, :, t]) # warmup in fresh process

elapsed = @elapsed begin
    a = Array(@view q[1:ztop, :, :, t])
    global mb = sizeof(a) / 1024.0^2
end

println(round(elapsed; digits=6), \",\", round(mb; digits=3))
"""

    cmd = `julia --startup-file=no --project=. -e $code`
    out = read(cmd, String)
    line = split(strip(out), '\n')[end]
    parts = split(line, ',')
    return parse(Float64, strip(parts[1])), parse(Float64, strip(parts[2]))
end

function main()
    # Discover nz once.
    probe = read(`julia --startup-file=no --project=. -e "include(\"src/MLCondensateDistributions.jl\"); using .MLCondensateDistributions; ds=MLCondensateDistributions.GoogleLES.load_zarr_simulation($SITE_ID,$MONTH,\"$EXPERIMENT\"); q=ds[\"q_c\"]; println(size(q)[1],\",\",size(q)[4])"`, String)
    p = split(strip(probe), ',')
    nz = parse(Int, p[1])
    nt = parse(Int, p[2])

    z_default = "1,2,4,8,16,32,64,128,256,320,$nz"
    z_list = [z for z in parse_int_list(get(ENV, "Z_LIST", z_default)) if 1 <= z <= nz]

    println("== profile_qc_z_window_scaling_cold ==")
    println("site=", SITE_ID, " month=", MONTH, " experiment=", EXPERIMENT, " repeats=", REPEATS)
    println("nz=", nz, " nt=", nt, " z_list=", z_list)
    println("z_layers,seconds_median,mb,ms_per_layer")

    for z in z_list
        times = Float64[]
        mb = 0.0
        for r in 1:REPEATS
            # Cycle timesteps to reduce reusing exactly same remote slice path.
            t_idx = 1 + ((r - 1) % max(nt, 1))
            sec, mb_cur = run_point(z, t_idx)
            push!(times, sec)
            mb = mb_cur
        end
        med = median_val(times)
        println(z, ",", round(med; digits=6), ",", round(mb; digits=3), ",", round(1000 * med / z; digits=4))
    end
end

main()
