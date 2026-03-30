# Training Dataset Specification

The extracted and coarse-grained data will adhere strictly to the following 1D flattened layout, generated natively. All spatial mapping coordinates (`x_idx`, `y_idx`) have been completely purged from the export matrix to strictly maintain horizontal translational invariance, leaving only altitude (`z_idx`) which inherently stratifies the base atmospheric vectors.

## Data Layout (Strict Column Order)

The output `DataFrame` and serialized Arrow object is guaranteed to be constructed with columns strictly mapping to the following definitions in this exact numerical index sequence:

| Column # | Code Name | Format | Origin | Description |
|---|---|---|---|---|
| **Means** |
| 1 | `qt` | Float32 | Physical | Total water specific humidity (`hus`) mean |
| 2 | `theta_li` | Float32 | Physical | Liquid-ice potential temperature (`thetali`) mean |
| 3 | `p` | Float32 | Physical | Hydrostatic or Hydrodynamic pressure (`pfull`) mean |
| 4 | `rho` | Float32 | Physical | Air density (`rhoa`) mean |
| 5 | `w` | Float32 | Physical | Vertical velocity (`wa`) mean |
| 6 | `q_liq` | Float32 | Physical | Liquid water specific humidity (`clw`) mean |
| 7 | `q_ice` | Float32 | Physical | Ice water specific humidity (`cli`) mean |
| 8 | `q_con` | Float32 | Derived | Total condensate mathematically combined as `clw` + `cli` mean |
| 9 | `tke` | Float32 | Derived  | Turbulence kinetic energy mathematically derived from `(ua, va, wa)` |
| **Variances** |
| 10 | `var_qt` | Float32 | Derived | Variance of total water |
| 11 | `var_ql` | Float32 | Derived | Variance of liquid water |
| 12 | `var_qi` | Float32 | Derived | Variance of ice water |
| 13 | `var_w` | Float32 | Derived | Variance of vertical velocity |
| 14 | `var_h` | Float32 | Derived | Variance of liquid-ice potential temperature |
| **Covariances** |
| 15 | `cov_qt_ql` | Float32 | Derived | Covariance of total water & liquid water |
| 16 | `cov_qt_qi` | Float32 | Derived | Covariance of total water & ice water |
| 17 | `cov_qt_w` | Float32 | Derived | Covariance of total water & vertical velocity |
| 18 | `cov_qt_h` | Float32 | Derived | Covariance of total water & liquid-ice potential temperature |
| 19 | `cov_ql_qi` | Float32 | Derived | Covariance of liquid water & ice water |
| 20 | `cov_ql_w` | Float32 | Derived | Covariance of liquid water & vertical velocity |
| 21 | `cov_ql_h` | Float32 | Derived | Covariance of liquid water & liquid-ice potential temperature |
| 22 | `cov_qi_w` | Float32 | Derived | Covariance of ice water & vertical velocity |
| 23 | `cov_qi_h` | Float32 | Derived | Covariance of ice water & liquid-ice potential temperature |
| 24 | `cov_w_h` | Float32 | Derived | Covariance of vertical velocity & liquid-ice potential temperature |
| **Scale Data** |
| 25 | `resolution_h` | Float32 | Metadata | Actual horizontal resolution in meters (e.g. `100.0`, `180.0`) |
| 26 | `resolution_z` | Float32 | Metadata | Actual vertical resolution in meters (e.g. `12.5`, `25.0`) |
| **Metadata** |
| 27 | `data_source` | String   | Metadata | Target environment (e.g. `GoogleLES`, `cfSite`) |
| 28 | `month` | Int64 | Metadata | Benchmark simulation month index parameter |
| 29 | `cfSite_number` | Int64 | Metadata | Internal Cloud Feedback Site map identification map ID |
| 30 | `forcing_model` | String | Metadata | Benchmark constraint boundary (e.g. `GFDL-CM4`) |
| 31 | `experiment` | String | Metadata | Experiment type (e.g. `amip`) |

## Validation
## Validation
Any testing pipelines or orchestration frameworks iterating over Arrow files output via `utils/dataset_builder.jl` are explicitly asserted against this specific 31-column map sequentially prior to payload serialization. If a specific structural variable array (like `thetali` or `rhoa`) is physically missing from the underlying dataset chunk, the pipeline is designed to throw a hard fatal error. This ensures that the generated training data is mathematically complete and never contains silent `NaN` or interpolated anomalies.
