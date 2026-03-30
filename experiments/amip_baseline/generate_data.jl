using Pkg; Pkg.activate(".")
using NN_Condensate_Distributions

# Experiment Workflow: Baseline AMIP Data Generation
# Site 10, Month 1, AMIP experiment.
# Generates a 5-timestep sample for initial research.

println("--- [Experiment: AMIP Baseline] Generating Data ---")

# The GoogleLES module now contains its own tabular builder
GoogleLES.build_tabular(
    10,                 # cfSite_number
    1,                  # month
    "amip",             # experiment_name
    "data/amip_baseline"; # local folder for this experiment
    max_timesteps=5
)

# Adding cfSites data generation as well
cfSites.build_tabular(
    10,                 # cfSite_number (Valid for HadGEM2-A, Month 01)
    1,                  # month
    "HadGEM2-A",        # forcing_model
    "amip",             # experiment_name
    "data/amip_baseline"; # local folder for this experiment
    max_timesteps=5
)

println("--- Data Generation Complete: data/amip_baseline ---")
