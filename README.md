# MLCondensateDistributions

MLCondensateDistributions builds coarse-grained LES datasets and trains neural-network surrogates for cloud condensate and related moments.

Current supported workflow focuses on GoogleLES AMIP data and produces Arrow datasets, Lux model artifacts, and diagnostics figures.

## Package Structure

- Core module: [src/MLCondensateDistributions.jl](src/MLCondensateDistributions.jl)
- Utilities: [utils](utils)
- Model definition: [src/models.jl](src/models.jl)
- Training pipeline: [utils/train_lux.jl](utils/train_lux.jl)
- AMIP scripts: [experiments/amip_baseline](experiments/amip_baseline)

## Current Training Design

Implemented in [utils/train_lux.jl](utils/train_lux.jl).

- Inputs are standardized.
- Targets are standardized.
- For phase targets (`q_liq`, `q_ice`), model output uses two channels per target:
    - magnitude channel (regression in standardized space),
    - presence channel (binary logit for zero vs nonzero).
- Training loss for phase targets combines:
    - BCE-with-logits on presence,
    - MSE on magnitude for present samples only.
- Non-phase targets (`var_*`, `cov_*`) use MSE in standardized space.
- Validation-based early stopping restores the best checkpoint before save.

Decode behavior:

- Magnitude is de-standardized to physical units.
- Presence probability threshold determines whether output is exact zero.
- Phase outputs are kept nonnegative after decode.

## Outputs

- Processed data: [data/processed](data/processed)
- Model artifacts: [data/models](data/models)
- AMIP diagnostics: [data/models/diagnostics_full/latest](data/models/diagnostics_full/latest)

## Running the AMIP Pipeline

Use [experiments/amip_baseline/README.md](experiments/amip_baseline/README.md) for the full workflow.

Typical REPL entrypoint:

```julia
using Pkg
Pkg.activate("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline")
include("/home/jbenjami/Research_Schneider/CliMA/MLCondensateDistributions/experiments/amip_baseline/full_train_experiment.jl")
run_full_training!(; make_plots=true)
```

## API Surface (Exported)

From [src/MLCondensateDistributions.jl](src/MLCondensateDistributions.jl):

- Modules: `Paths`, `WorkflowState`, `Dynamics`, `CoarseGraining`, `DatasetBuilder`, `GoogleLES`, `cfSites`
- Functions/types: `CondensateMLP`, `load_processed_data`, `standardize_data`, `train_model`, `write_training_diagnostics`

## Notes

- The diagnostics plotting hook is extension-based and provided by [ext/CairoMakieExt.jl](ext/CairoMakieExt.jl) when CairoMakie is available.
- Documentation for lower-level utilities is in [utils/README.md](utils/README.md).
- How GoogleLES `.arrow` generation works (z-mask, per-span Zarr loads, materialization): [docs/googleles_build_tabular.md](docs/googleles_build_tabular.md).
- GoogleLES Zarr `size` / `chunks` / `_ARRAY_DIMENSIONS` axis conventions (480=z, 73=t, permutation rule): [docs/googleles_zarr_layout.md](docs/googleles_zarr_layout.md).
