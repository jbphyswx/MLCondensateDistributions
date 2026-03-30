"""
    Python version for reading from the Google Cloud cause idk if Julia has the best tools for it yet.

    See: https://github.com/google-research/swirl-lm/blob/main/swirl_lm/example/geo_flows/cloud_feedback/README.md for more info on the data.

    # ------------------------------------------------------------------- #
    # KEY VARIABLES METADATA (from Swirl-LM documentation)
    # ------------------------------------------------------------------- #
    # 3D Variables (t, x, y, z):
    #   T              : Air temperature (K)
    #   q_t            : Total water specific humidity (kg/kg)
    #   q_c            : Condensed phase specific humidity (kg/kg)
    #   q_r            : Rain droplet mass fraction (kg/kg)
    #   q_s            : Snow mass fraction (kg/kg)
    #   u, v, w        : Zonal, Meridional, and Vertical velocities (m/s)
    #   p, p_ref       : Hydrodynamic & Reference pressures (Pa)
    #   rho            : Air density (kg/m^3)
    #   theta_li       : Liquid-ice potential temperature (K)
    #   rad_flux_lw/sw : Net longwave/shortwave radiative fluxes (W/m^2)
    # 
    # 1D/2D Variables:
    #   lwp, iwp       : Liquid & Ice water paths (kg/m^2)
    #   cloud_cover    : Column cloud fraction (unitless)
    #   cloud_fraction : Profile cloud fraction (t, z) (unitless)
    #   cre_lw, cre_sw : Longwave & Shortwave Cloud Radiative Effects (W/m^2)
    #   olr, asr       : Outgoing Longwave & Absorbed Shortwave radiation at TOA
    # ------------------------------------------------------------------- #

    Bulk statistics: https://console.cloud.google.com/storage/browser/cloudbench-statistics;tab=objects?prefix=&forceOnObjectsSortingFiltering=false (not what we need but useful to have)


    Full output: https://console.cloud.google.com/storage/browser/cloudbench-simulation-output;tab=objects?prefix=&forceOnObjectsSortingFiltering=false (500 TB, so we need to stream data)

    Note: You must have a Conda environment with `xarray`, `zarr`, `gcsfs`, and `h5netcdf` installed to run these functions.
"""

import xarray as xr

RAW_OUTPUT_BUCKET = "gs://cloudbench-simulation-output"
STATS_BUCKET = "gs://cloudbench-statistics"

def load_zarr_simulation(site_id: int, month: int, experiment: str = "amip"):
    """
    Lazily loads the raw 3D zarr output for a given site, month, and experiment.
    Since the dataset is 500 TB, this ONLY fetches metadata initially.
    
    Args:
        site_id (int): Location index (0 to 499)
        month (int): Simulated month (1, 4, 7, 10)
        experiment (str): The configuration (e.g. 'amip', 'amip-p4k', 'amip-4xco2')
        
    Returns:
        xarray.Dataset: The lazy Zarr dataset.
    """
    path = f"{RAW_OUTPUT_BUCKET}/{site_id}/{month}/{experiment}/data.zarr"
    print(f"Loading metadata from {path}...")
    # `anon=True` allows unauthenticated access to public buckets via gcsfs
    ds = xr.open_zarr(path, storage_options={"anon": True})
    return ds

def load_statistics(filename="cloudbench_statistics.nc"):
    """
    Loads the post-processed consolidated statistics netCDF file over HTTP.
    
    Args:
        filename (str): Name of the consolidated NetCDF file in the statistics bucket.
    """
    import fsspec
    
    path = f"{STATS_BUCKET}/{filename}"
    print(f"Streaming NetCDF over GCS from {path}...")
    
    try:
        # Use gcsfs to open the remote file, then read it into xarray
        fs = fsspec.filesystem("gcs", anon=True)
        remote_file = fs.open(path, "rb")
        # For remote NetCDF via file-like objects, the h5netcdf engine handles chunking natively.
        ds = xr.open_dataset(remote_file, engine="h5netcdf")
        return ds
    except Exception as e:
        print(f"Failed to load {path}. Error: {e}")
        return None