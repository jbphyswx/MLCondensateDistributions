#=
    Planned closures for scorecards vs tabular truth or NN outputs. Implement from the cited papers;
    TurbulenceConvection.jl (paths below) is an optional coding reference only, not a dependency.

    Full implementations and model scorecards are intentionally deferred until EDA in
    `experiments/analyze_truth_data/scripts/` shows which moments and regimes to match.
=#

"""
    See /home/jbenjami/Research_Schneider/CliMA/TurbulenceConvection.jl/src/closures/sgs_condensate/SHOC/SHOC.jl (but do not assume it is correct or complete!)

    shoc_assumed_pdf_point(
        thl_mean, qt_mean, w_mean, w2, w3,
        thl_var, qt_var, thl_qt_cov, wthl, wqw,
        p, thermo_params; dothetal_skew = false, liq_frac_mean = 1,
    )

Compute SHOC Analytic Double-Gaussian (ADG1) cloud statistics and liquid-water flux
for a single level using the full SHOC closure.

Algorithm:
1. Computes bimodal vertical velocity distribution (w1, w2) based on w2 and w3 moments
2. Computes bimodal θl and qt distributions for each velocity mode
3. Computes saturation using mixed-phase saturation: qs = λ*qs_liq + (1-λ)*qs_ice
4. Computes saturation deficit and cloud fraction per plume using ADG1 formulation
5. Computes area-weighted cloud fraction and condensate amount

Important notes:
- **ql_mean is TOTAL CONDENSATE (liquid + ice), not just liquid!** 
  The function computes condensate using mixed-phase saturation based on liq_frac_mean.
  Caller must partition into liquid/ice using liquid_fraction if needed.
- This uses saturation mixing ratio and the ADG1 saturation-deficit formulation; it is
  not a full saturation-adjustment step.
- The closure is identical for Equilibrium and NonEquilibrium moisture models; any
  phase partitioning is handled outside this function.
- `liq_frac_mean` is a linearized liquid fraction (0..1) used to blend liquid/ice
  saturation in the plume thermodynamics.

Returns:
- cloud_frac: Area-weighted cloud fraction (0..1)
- ql_mean: Mean TOTAL condensate (liquid + ice) [kg/kg]
- wqls: Vertical flux of liquid water (should be partitioned to ql/qi by caller)
- wthv_sec: Virtual potential temperature flux (buoyancy flux)
- ql_var: Variance of liquid water
- s1: Signed saturation deficit in plume 1 (can be negative for undersaturation)
- s2: Signed saturation deficit in plume 2 (can be negative for undersaturation)
- thl1_1, thl1_2: Liquid potential temperature in plumes 1a and 1b
- qw1_1, qw1_2: Total water content in plumes 1a and 1b
- w1_1, w1_2: Vertical velocity in plumes 1a and 1b
- a: Bimodal area fraction
- qs1, qs2: Saturation mixing ratio in plumes 1a and 1b

Reference: Bogenschutz & Krueger (2013), ADG1 closure
"""
function SHOC()
    error("SHOC not implemented yet")
end

""" These are what underly SHOC but i think these versions are prognostic so idk if we can use them """
function ADGC1()
    error("ADGC1 not implemented yet")
end

""" These are what underly SHOC but i think these versions are prognostic so idk if we can use them """
function ADGC2()
    error("ADGC2 not implemented yet")
end

"""
    See /home/jbenjami/Research_Schneider/CliMA/TurbulenceConvection.jl/src/closures/sgs_condensate/cSigma/cSigma.jl (but do not assume it is correct or complete!)

    --------------------------------------------------------------------------------
    METHODOLOGY: The "cSigma" Parameterization
    Reference: Larson et al. (2011), "Parameterizing correlations between hydrometeor 
    species in mixed-phase Arctic clouds", J. Geophys. Res.

    THE PROBLEM:
    Microphysical processes (like accretion) depend heavily on the subgrid correlation 
    between species. Arbitrarily defining these correlations often leads to "impossible" 
    matrices that are not positive semidefinite.

    THE SOLUTION:
    1. Spherical Parameterization: We predict the Cholesky factor (L) of the correlation 
    matrix (Sigma = L * L'), guaranteeing positive semidefiniteness.

    2. "cSigma" Closure: We estimate the angles of L using:
    a) Known correlations with vertical velocity (w).
    b) A closure based on subgrid variability (std/mean).

    c_ij = c_1i * c_1j + f_ij * s_1i * s_1j   [Eq. 15]
    f_ij = alpha * S_i * S_j * sgn(c_1i * c_1j)  [Eq. 16]

    EXTENSION (Prognostic Covariances):
    If specific covariances (e.g., H-QT) are known/prognosed, they can be passed in 
    to override the cSigma prediction for those specific elements.
    --------------------------------------------------------------------------------
"""
function cSigma()
    error("cSigma not implemented yet")
end