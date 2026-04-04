# Training Dataset Specification

The extracted and coarse-grained data will adhere strictly to the following 1D flattened layout, generated natively. All spatial mapping coordinates (`x_idx`, `y_idx`) have been completely purged from the export matrix to strictly maintain horizontal translational invariance, leaving only altitude (`z_idx`) which inherently stratifies the base atmospheric vectors.

## Data Layout (Strict Column Order)

The output `DataFrame` and serialized Arrow object is guaranteed to be constructed with columns strictly mapping to the following definitions in this exact numerical index sequence:

| Column # | Code Name | Format | Origin | Description |
|---|---|---|---|---|
| **Means** |
| 1 | `qt` | Float32 | Physical | Total water specific humidity (`hus`) mean |
| 2 | `theta_li` | Float32 | Physical | Liquid-ice potential temperature (`thetali`) mean |
| 3 | `ta` | Float32 | Physical | Absolute temperature in Kelvin (`ta`) mean |
| 4 | `p` | Float32 | Physical | Hydrostatic or Hydrodynamic pressure (`pfull`) mean |
| 5 | `rho` | Float32 | Physical | Air density (`rhoa`) mean |
| 6 | `w` | Float32 | Physical | Vertical velocity (`wa`) mean |
| 7 | `q_liq` | Float32 | Physical | Liquid water specific humidity (`clw`) mean |
| 8 | `q_ice` | Float32 | Physical | Ice water specific humidity (`cli`) mean |
| 9 | `q_con` | Float32 | Derived | Total condensate mathematically combined as `clw` + `cli` mean |
| 10 | `liq_fraction` | Float32 | Derived | Coarse-cell liquid area fraction, computed as horizontal average of `1(clw > 1e-10)` |
| 11 | `ice_fraction` | Float32 | Derived | Coarse-cell ice area fraction, computed as horizontal average of `1(cli > 1e-10)` |
| 12 | `cloud_fraction` | Float32 | Derived | Coarse-cell cloud area fraction, computed as horizontal average of `1((clw + cli) > 1e-10)` |
| 13 | `tke` | Float32 | Derived  | Turbulence kinetic energy mathematically derived from `(ua, va, wa)` |
| **Variances** |
| 14 | `var_qt` | Float32 | Derived | Variance of total water |
| 15 | `var_ql` | Float32 | Derived | Variance of liquid water |
| 16 | `var_qi` | Float32 | Derived | Variance of ice water |
| 17 | `var_w` | Float32 | Derived | Variance of vertical velocity |
| 18 | `var_h` | Float32 | Derived | Variance of liquid-ice potential temperature |
| **Covariances** |
| 19 | `cov_qt_ql` | Float32 | Derived | Covariance of total water & liquid water |
| 20 | `cov_qt_qi` | Float32 | Derived | Covariance of total water & ice water |
| 21 | `cov_qt_w` | Float32 | Derived | Covariance of total water & vertical velocity |
| 22 | `cov_qt_h` | Float32 | Derived | Covariance of total water & liquid-ice potential temperature |
| 23 | `cov_ql_qi` | Float32 | Derived | Covariance of liquid water & ice water |
| 24 | `cov_ql_w` | Float32 | Derived | Covariance of liquid water & vertical velocity |
| 25 | `cov_ql_h` | Float32 | Derived | Covariance of liquid water & liquid-ice potential temperature |
| 26 | `cov_qi_w` | Float32 | Derived | Covariance of ice water & vertical velocity |
| 27 | `cov_qi_h` | Float32 | Derived | Covariance of ice water & liquid-ice potential temperature |
| 28 | `cov_w_h` | Float32 | Derived | Covariance of vertical velocity & liquid-ice potential temperature |
| **Scale Data** |
| 29 | `resolution_h` | Float32 | Metadata | Horizontal grid spacing in meters after coarse-graining (e.g. `100.0`, `180.0`) |
| 30 | `domain_h` | Float32 | Metadata | Horizontal domain span in meters covered by the LES case (e.g. `6000.0`) |
| 31 | `resolution_z` | Float32 | Metadata | Actual vertical resolution in meters (e.g. `12.5`, `25.0`) |
| **Metadata** |
| 32 | `data_source` | String   | Metadata | Target environment (e.g. `GoogleLES`, `cfSite`) |
| 33 | `month` | Int64 | Metadata | Benchmark simulation month index parameter |
| 34 | `cfSite_number` | Int64 | Metadata | Internal Cloud Feedback Site map identification map ID |
| 35 | `forcing_model` | String | Metadata | Benchmark constraint boundary (e.g. `GFDL-CM4`) |
| 36 | `experiment` | String | Metadata | Experiment type (e.g. `amip`) |

## Validation
Any testing pipelines or orchestration frameworks iterating over Arrow files output via `utils/dataset_builder.jl` are explicitly asserted against this specific 36-column map sequentially prior to payload serialization. If a specific structural variable array (like `ta`, `thetali` or `rhoa`) is physically missing from the underlying dataset chunk, the pipeline is designed to throw a hard fatal error. This ensures that the generated training data is mathematically complete and never contains silent `NaN` or interpolated anomalies.
