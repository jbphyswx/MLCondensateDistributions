module MLCondensateDistributions

using Lux: Lux
using Random: Random

# 1. Physical Utilities (Fundamental building blocks)
include("../utils/dynamics.jl")
include("../utils/coarse_graining.jl")

# 2. Dataset Logic Hub (Used by data source modules)
include("../utils/dataset_builder.jl")

# 3. Data Source Modules (Provide source-specific loaders and 'build_tabular' methods)
include("../utils/GoogleLES.jl")
include("../utils/cfSites.jl")

# 4. Training and Models
include("../utils/build_training_data.jl")
include("../utils/dataloader.jl")
include("models.jl")

# Export submodules for use in experiments
export Dynamics, CoarseGraining, DatasetBuilder, GoogleLES, cfSites
export CondensateMLP, load_processed_data, standardize_data

end # module
