This is meant to be where we can train a neural network to predict the probability distribution of liquid and ice water content in a cloud.


We will train from limited LES data (which is all probably with temperature based partitioning right now).

First, we hae Zhaoyi's GCM-froced LES data.
To use: 

Then, there's the Google LES data.
To use:



The goal is to be able to predict on a quadrature grid, the amounts of condensate in each cell.



So you can go straight to the quadrature bins, i.e:

    [q_liq_quad, q_ice_quad] = func(qt_mean, qt_var, qt_hat (or qt_SD), H_mean, h_var, h_hat (or h_SD), q_liq, q_ice, tau_liq, tau_ice, w, tke, res)
    

for noneq. (Is there a way we can enforce normalization of the output distribution)
As a first pass it doesnt contain other outputs like N_ice, but it could be made to.. for now just scale those equally if you need to (despite things like INP temperature scaling etc). We should also calcluate CF_mp as a function of inputs as an exercise.
Ideally we will compare the network outputs to things like Furtado's limiting regimes.

We should downscale the LES data at separate resolutions to train the res parameter. Idk if res should include vertical but probably... idk.

The network itself would just predict q_liq, q_ice at any point. 
The end user would integrate over the bins (either at bin centers or do bin edges then integrate), and normalize to match an existing q_c for example.
It would be nice to get direct to bin predictions but variable output sizes are hard.


For eq, maybe we try to predict the liquid fraction in each bin idk.

We could also then have a separate fucntion that tried to predict covariances between:

(w, qt), (w, ql), (w, qi), (w,h)
(ql, qt), (ql, h)
(qi, qt), (qi, h)

in eq and noneq again.

TBD on using SimpleChains or Lux or Flux


Finally we'll run some SCM sims with and without the network and compare to the LES...
For right now there are no plans to online optimiz the network.



Since in principle this might be very profile dependent, we'd like to build a transformer fno version that one shots the entire profile together, using attention mechanisms to ensure commmunication between vertical levels.


Finally we expect as resolution drops to see a handoff between solely local turbulence and grid scale gradient driven variance.
Maybe it would have been good to pass in dT/dz and d\theta_l/dz as well, but those probably become more crucial at lower resolutions, i wonder if we can show that.


For train test split we could due current and then future climate (4K or whatever), or split by month, etc... tbd

## NetCDF Data Structure Notes

### MPI Chunking (Fields)
The raw 3D LES fields are stored in MPI chunks (`0.nc`, `1.nc`, etc.). 
- **Global Grid**: Metadata for global dimensions is in the `dims` group under `x`, `y`, `z` (and also variables `n_0`, `n_1`, `n_2`).
- **Local Grid**: Rank-specific local dimensions are stored as `nl_0`, `nl_1`, `nl_2`.
- **Placement**: Offset metadata for each chunk is in `indx_lo_0`, `indx_lo_1`, `indx_lo_2`.
- **Note**: When loading fields, use `nl_*` for reshaping chunks to avoid `DimensionMismatch`.

### Stats Files
- **Reference Pressure (p0)**: Usually found in the `reference` group (e.g., `reference/p0` or `reference/p0_full`). 
- **Profiles**: 1D time-averaged stats are in the `profiles` group.
