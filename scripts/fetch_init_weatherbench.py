#code to pull initialisation file from weatherbench

import math
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--start", required=True, help="Initialisation start time, e.g. 2021-06-18T00:00:00")
parser.add_argument("--forecast-days", required=True, type=int, help="Number of forecast days *after* initialisation")
parser.add_argument("--output-file", required=True, help="NetCDF output file")
args = parser.parse_args()
start_dt = datetime.fromisoformat(args.start)

timestep = 12 #hours
forecast_steps = round(args.forecast_days * 24 / timestep)
total_steps = forecast_steps + 2
end_dt = start_dt + timedelta(hours=(total_steps - 1) * timestep)

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def rename_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    rename_map = {}
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"
    return ds.rename(rename_map) if rename_map else ds


def add_graphcast_progress_features(ds: xr.Dataset, start: str) -> xr.Dataset:
    """
    Add only:
      year_progress_sin -> (batch, time)
      year_progress_cos -> (batch, time)
      day_progress_sin  -> (batch, time, lon)
      day_progress_cos  -> (batch, time, lon)

    The raw year_progress/day_progress scalars are not kept.
    """
    if "time" not in ds.coords:
        raise ValueError("Expected a time coordinate.")
    if "lon" not in ds.coords:
        raise ValueError("Expected a lon coordinate.")

    # time is already rebased to integer hours relative to START
    hours = ds["time"].values.astype(np.int64)
    dt = np.datetime64(start) + hours.astype("timedelta64[h]")
    seconds_since_epoch = dt.astype("datetime64[s]").astype(np.int64)

    year_phase_1d = np.mod(
        seconds_since_epoch / (86400.0 * 365.24219),
        1.0,
    ).astype(np.float32)

    lon = ds["lon"].values.astype(np.float32)
    day_greenwich = np.mod(seconds_since_epoch, 86400) / 86400.0
    day_phase_2d = np.mod(
        day_greenwich[:, None] + lon[None, :] / 360.0,
        1.0,
    ).astype(np.float32)

    if "batch" in ds.dims:
        batch_size = ds.sizes["batch"]
        batch_vals = ds["batch"].values if "batch" in ds.coords else np.arange(batch_size, dtype=np.int64)

        year_phase = np.broadcast_to(
            year_phase_1d[None, :],
            (batch_size, year_phase_1d.shape[0]),
        ).astype(np.float32)

        day_phase = np.broadcast_to(
            day_phase_2d[None, :, :],
            (batch_size, day_phase_2d.shape[0], day_phase_2d.shape[1]),
        ).astype(np.float32)

        ds["year_progress_sin"] = xr.DataArray(
            np.sin(2 * np.pi * year_phase).astype(np.float32),
            dims=("batch", "time"),
            coords={"batch": batch_vals, "time": ds["time"]},
        )
        ds["year_progress_cos"] = xr.DataArray(
            np.cos(2 * np.pi * year_phase).astype(np.float32),
            dims=("batch", "time"),
            coords={"batch": batch_vals, "time": ds["time"]},
        )
        ds["day_progress_sin"] = xr.DataArray(
            np.sin(2 * np.pi * day_phase).astype(np.float32),
            dims=("batch", "time", "lon"),
            coords={"batch": batch_vals, "time": ds["time"], "lon": ds["lon"]},
        )
        ds["day_progress_cos"] = xr.DataArray(
            np.cos(2 * np.pi * day_phase).astype(np.float32),
            dims=("batch", "time", "lon"),
            coords={"batch": batch_vals, "time": ds["time"], "lon": ds["lon"]},
        )
    else:
        ds["year_progress_sin"] = xr.DataArray(
            np.sin(2 * np.pi * year_phase_1d).astype(np.float32),
            dims=("time",),
            coords={"time": ds["time"]},
        )
        ds["year_progress_cos"] = xr.DataArray(
            np.cos(2 * np.pi * year_phase_1d).astype(np.float32),
            dims=("time",),
            coords={"time": ds["time"]},
        )
        ds["day_progress_sin"] = xr.DataArray(
            np.sin(2 * np.pi * day_phase_2d).astype(np.float32),
            dims=("time", "lon"),
            coords={"time": ds["time"], "lon": ds["lon"]},
        )
        ds["day_progress_cos"] = xr.DataArray(
            np.cos(2 * np.pi * day_phase_2d).astype(np.float32),
            dims=("time", "lon"),
            coords={"time": ds["time"], "lon": ds["lon"]},
        )

    return ds

PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"

# open lazily
ds = xr.open_zarr(
    PATH,
    consolidated=True,
    storage_options={"token": "anon"},
    chunks="auto",
)

OUTPUT_FILE = args.output_file
START = args.start  # "2021-06-18T00:00:00"
END = end_dt.isoformat()  # "2021-07-03T12:00:00"

REQUESTED = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "sst": "sea_surface_temperature",
    "lsm": "land_sea_mask",
    "zs": "geopotential_at_surface",
}

ORDERED_VARS = [
    "land_sea_mask",
    "geopotential_at_surface",
    "day_progress_cos",
    "day_progress_sin",
    "2m_temperature",
    "sea_surface_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "total_precipitation_12hr",
    "10m_u_component_of_wind",
    "u_component_of_wind",
    "specific_humidity",
    "temperature",
    "vertical_velocity",
    "v_component_of_wind",
    "geopotential",
    "year_progress_cos",
    "year_progress_sin",
]

# -------------------------------------------------------------
# VARIABLE SELECTION
# -------------------------------------------------------------
present = {k: v for k, v in REQUESTED.items() if v in ds.data_vars}
missing = {k: v for k, v in REQUESTED.items() if v not in ds.data_vars}

print("Present:", present)
print("Missing:", missing)

if not present:
    raise ValueError("None of the requested variables are present in the dataset.")

ds = ds[list(present.values())]

# -------------------------------------------------------------
# COORDS / TIME WINDOW
# -------------------------------------------------------------
ds = rename_lat_lon(ds)
if "lat" not in ds.dims or "lon" not in ds.dims:
    raise ValueError("Expected latitude/longitude dimensions after renaming.")

ds = ds.sel(time=slice(START, END))
ds = ds.sel(time=ds.time.dt.hour.isin([0, timestep]))

#if ds.sizes["time"] < 32:
#    raise ValueError(f"Expected at least 32 timesteps, found {ds.sizes['time']}.")

ds = ds.isel(time=slice(0, 32))

# Rebase time to 0, 12, 24, ..., 372
time_hours = ((ds["time"].values - ds["time"].values[0]) / np.timedelta64(1, "h")).astype(np.int32)
#if time_hours.size != 32:
#    raise ValueError("Time rebase failed; expected 32 hourly offsets.")

ds = ds.assign_coords(time=("time", time_hours))
ds["time"].attrs = {"units": "hours"}

# Keep level dtype aligned with reference
if "level" in ds.coords:
    ds = ds.assign_coords(level=ds["level"].astype(np.int32))

# -------------------------------------------------------------
# REGRID
# -------------------------------------------------------------
target_lat = np.linspace(-90.0, 90.0, 181, dtype=np.float32)
target_lon = np.linspace(0.0, 359.0, 360, dtype=np.float32)
ds = ds.interp(lat=target_lat, lon=target_lon, method="linear")

# -------------------------------------------------------------
# BATCH
# -------------------------------------------------------------
if "batch" not in ds.dims:
    ds = ds.expand_dims(batch=[0])

# -------------------------------------------------------------
# PRECIPITATION
# -------------------------------------------------------------
if "total_precipitation_12hr" not in ds.data_vars:
    ds["total_precipitation_12hr"] = xr.zeros_like(ds["2m_temperature"])

# -------------------------------------------------------------
# GRAPHCAST TIME FEATURES
# -------------------------------------------------------------
ds = add_graphcast_progress_features(ds, START)

# Raw progress scalars should not be in the saved file
ds = ds.drop_vars([v for v in ("year_progress", "day_progress") if v in ds.data_vars], errors="ignore")

# Final datetime coordinate should be a batched hour-offset coordinate
datetime_hours = np.broadcast_to(
    time_hours[None, :],
    (ds.sizes["batch"], ds.sizes["time"]),
).astype(np.int32)
ds = ds.assign_coords(datetime=(("batch", "time"), datetime_hours))
ds["datetime"].attrs = {"units": f"hours since {START.replace('T', ' ')}"}

# -------------------------------------------------------------
# STATIC FIELDS SHOULD STAY 2D
# -------------------------------------------------------------
for static_name in ("land_sea_mask", "geopotential_at_surface"):
    if static_name in ds.data_vars and "batch" in ds[static_name].dims:
        ds[static_name] = ds[static_name].isel(batch=0, drop=True)

# -------------------------------------------------------------
# FINAL CHECK + SAVE
# -------------------------------------------------------------
#ds = ds.load()
ds.to_netcdf(OUTPUT_FILE, format="NETCDF4")
print(f"Saved to {OUTPUT_FILE}")

#like 47mins

print("\n================ DATASET SUMMARY ================\n")

print("Sizes:")
print(dict(ds.sizes))

print("\nCoordinates:")
for name, coord in ds.coords.items():
    print(f"{name:25s} {coord.dims} {coord.dtype}")

print("\nVariables:")
for name, da in ds.data_vars.items():
    print(f"{name:30s} {da.dims} {da.dtype}")

print("\nProgress variables:")
print([v for v in ds.data_vars if "progress" in v])

print("\nStatic variables:")
for v in ["land_sea_mask", "geopotential_at_surface"]:
    if v in ds:
        print(f"{v}: {ds[v].dims}")

print("\nLevel dtype:")
print(ds["level"].dtype)

print("\nDatetime:")
print(ds["datetime"].dims)
print(ds["datetime"].dtype)

print("\nTime values:")
print(ds["time"].values)

print("\n=================================================\n")
