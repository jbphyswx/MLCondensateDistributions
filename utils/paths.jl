"""
Shared path helpers for MLCondensateDistributions.

These functions keep generated data in one package-level location instead of
scattering experiment outputs across multiple subtrees.
"""
module Paths

export package_root, package_data_root, processed_data_root, tests_data_root, model_data_root, workflow_cache_root, experiment_data_dir, test_data_dir, model_data_dir

"""Return the package root directory containing `Project.toml`."""
package_root() = normpath(joinpath(@__DIR__, ".."))

"""Return the package-wide generated data root."""
package_data_root() = joinpath(package_root(), "data")

"""Return the shared processed-data directory used by training workflows."""
processed_data_root() = joinpath(package_data_root(), "processed")

"""Return the storage directory for test workflows."""
tests_data_root() = joinpath(package_data_root(), "tests")

"""Return the shared directory used for trained model artifacts."""
model_data_root() = joinpath(package_data_root(), "models")

"""Return the directory for workflow bookkeeping such as checked-simulation manifests."""
workflow_cache_root() = joinpath(package_data_root(), "workflow_cache")

"""Backward-compatible alias for the shared processed-data directory."""
experiment_data_dir(name::AbstractString) = processed_data_root()

"""Return a test-specific folder under the shared tests root."""
test_data_dir(name::AbstractString) = joinpath(tests_data_root(), name)

"""Backward-compatible alias for the shared models directory."""
model_data_dir() = model_data_root()

end # module