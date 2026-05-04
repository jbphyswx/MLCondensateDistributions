"""
MLCondensateDistributions

Package for generating coarse-grained LES training data and training Lux models
for condensate and moment targets.

Main exported surfaces:
- Data/build modules: `Paths`, `WorkflowState`, `Dynamics`, `StatisticalMethods`,
  `CoarseGraining`, `DatasetBuilder`, `GoogleLES`, `cfSites`.
- Training/model APIs: `CondensateMLP`, `load_processed_data`,
  `standardize_data`, `train_model`.
- Optional diagnostics hook: `write_training_diagnostics`.

GoogleLES Zarr axis conventions (`size`/`chunks` vs `_ARRAY_DIMENSIONS`): see `docs/googleles_zarr_layout.md`.
"""
module MLCondensateDistributions

using Lux: Lux
using Random: Random
using Dates: Dates

const MLCondensateDistributionsDefaults = (;
  
)

# 1. Physical Utilities (Fundamental building blocks)
include("../utils/paths.jl")
include("../utils/workflow_state.jl")
include("../utils/env_helpers.jl")
include("../utils/dynamics.jl")
include("../utils/statistical_methods/StatisticalMethods.jl")
using .StatisticalMethods: StatisticalMethods
include("../utils/coarse_graining.jl")
include("../utils/analysis.jl")

# 2. Dataset Logic Hub (Used by data source modules)
include("../utils/dataset_builder.jl")
using .DatasetBuilder: ReductionSpecs

# 3. Data Source Modules (Provide source-specific loaders and 'build_tabular' methods)
include("../utils/GoogleLES.jl")
include("../utils/cfSites.jl")

# 4. Training and Models
include("../utils/build_training_common.jl")
include("../utils/build_training_data.jl")
include("../utils/dataloader.jl")
include("../utils/data_handling.jl")
include("models.jl")
include("../utils/train_lux.jl")

# 5. Common utilities (generic data/I/O operations)
include("../utils/common.jl")

# 6. Visualization helpers
include("../viz/viz.jl")

"""
	write_training_diagnostics(args...)

Extension hook for optional plotting diagnostics.
By default this is a no-op and returns `nothing`.
When CairoMakie is available and the extension is loaded, a more specific
method is provided by the extension.
"""
write_training_diagnostics(args...) = nothing

# Export submodules for use in experiments
export Paths, WorkflowState, Dynamics, StatisticalMethods, CoarseGraining, DatasetBuilder, GoogleLES, cfSites
export TabularBuildOptions, tabular_build_options_from_env, tabular_options_with, tabular_build_options_summary
export ReductionSpecs
export DataHandling, Analysis, EnvHelpers, Common
export Viz
export CondensateMLP, load_processed_data, standardize_data, train_model
export write_training_diagnostics

end # module
