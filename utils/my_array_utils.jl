#=
    Scratch file — **not** loaded by `MLCondensateDistributions.jl`.

    Production coarsening APIs live in:
    - `utils/array_utils.jl` (`ArrayUtils`): kernels (`conv3d_block_mean`, `coarsen_dz_profile_factor`, …)
    - `utils/coarsening_pipeline.jl` (`CoarseningPipeline`): schedules and named-tuple coarsen
      (`build_convolutional_coarsening_triples`, `coarsen_fields_3d_block`, `coarsen_products_3d_block`, …)
    - `utils/dataset_builder_impl.jl`: `spatial_info.coarsening_mode` is `:hybrid`, `:block`, or `:sliding`.

    To experiment in a script:
        include(joinpath(path_to_pkg, "utils", "coarsening_pipeline.jl"))
        using .CoarseningPipeline
=#
