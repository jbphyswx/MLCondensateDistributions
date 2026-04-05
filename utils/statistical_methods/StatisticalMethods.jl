"""
    StatisticalMethods

Parallel merge of variance/covariance sufficient statistics (Chan, Pebay) and helpers for
per-voxel covariance from coarse means (`covariance_from_moments`, `covariance_from_moments!`).

Import with an alias when you want a short name, e.g.
`using MLCondensateDistributions: StatisticalMethods as SM`, or `import MLCondensateDistributions as MLCD` and
then `MLCD.StatisticalMethods`. If you `include` the package `src` file into `Main` instead of loading
via `Pkg`, use a leading dot: `using .MLCondensateDistributions: StatisticalMethods as SM`.

See [`docs/MOMENTS_NUMERICS_PIPELINE.md`](@ref).
"""
module StatisticalMethods

include("parallel_merge.jl")
include("covariance_fields.jl")

export merge_variance_chan,
    merge_covariance_pebay,
    merge_variance_children,
    merge_covariance_children,
    covariance_from_moments,
    covariance_from_moments!

end
