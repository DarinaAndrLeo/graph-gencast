from pathlib import Path
import xarray as xr
import numpy as np
import tempfile
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--init-file", required=True, help="Initialisation NetCDF file")
parser.add_argument("--pi-or-incr", required=True, type=int, help="-1 for pi, 1 for incr")
parser.add_argument("--output-file", required=True, help="NetCDF output file")
args = parser.parse_args()

if args.pi_or_incr == 1:
    grib_path = "incr-co2-sstdelta-remap.grib"
elif args.pi_or_incr == -1:
    grib_path = "pi-co2-sstdelta-remap.grib"
else:
    raise ValueError("pi_or_incr must be -1 or 1")

init_nc_path=args.init_file
out_nc_path=args.output_file

def remap_grib_and_apply_to_sst(
    grib_path,
    init_nc_path,
    out_nc_path,
    target_var="sea_surface_temperature",
    cdo_bin="cdo",
):
    """
    Regularise a reduced-Gaussian GRIB with CDO, remap it to the target
    NetCDF lon/lat grid, verify the remap worked, and apply the delta to SST.

    Parameters
    ----------
    grib_path : str or Path
        Input GRIB file (e.g. pi-co2-sstdelta-remap.grib)
    init_nc_path : str or Path
        Target initialisation NetCDF (e.g. init_2021-06-25T12_10d.nc)
    out_nc_path : str or Path
        Output NetCDF to write
    target_var : str
        Variable in the NetCDF to add the delta to
    cdo_bin : str
        CDO executable name/path

    Returns
    -------
    out_ds : xarray.Dataset
        Dataset after delta application
    remap_ds : xarray.Dataset
        The remapped delta dataset opened from CDO output
    """
    grib_path = Path(grib_path)
    init_nc_path = Path(init_nc_path)
    out_nc_path = Path(out_nc_path)

    # Load target dataset
    init_ds = xr.open_dataset(init_nc_path)
    if target_var not in init_ds:
        raise KeyError(f"{target_var!r} not found in {init_nc_path}")

    if "lat" not in init_ds.coords or "lon" not in init_ds.coords:
        raise ValueError("Target NetCDF must have 'lat' and 'lon' coordinates")

    lat = init_ds["lat"].values
    lon = init_ds["lon"].values

    # Build a CDO grid description file for the target grid
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        gridfile = tmpdir / "target_grid.txt"
        regular_grib = tmpdir / "regularised.grib"
        remap_nc = tmpdir / "remapped.nc"

        with gridfile.open("w") as f:
            f.write("gridtype = lonlat\n")
            f.write(f"xsize    = {len(lon)}\n")
            f.write(f"ysize    = {len(lat)}\n")
            f.write(f"xfirst   = {float(lon[0])}\n")
            f.write(f"xinc     = {float(lon[1] - lon[0])}\n")
            f.write(f"yfirst   = {float(lat[0])}\n")
            f.write(f"yinc     = {float(lat[1] - lat[0])}\n")

        # 1) Regularise reduced Gaussian grid
        cmd1 = [cdo_bin, "setgridtype,regular", str(grib_path), str(regular_grib)]
        subprocess.run(cmd1, check=True)

        # 2) Remap onto target lon/lat grid
        cmd2 = [cdo_bin, "-f", "nc", f"remapbil,{gridfile}", str(regular_grib), str(remap_nc)]
        subprocess.run(cmd2, check=True)

        # Re-open remapped output
        remap_ds = xr.open_dataset(remap_nc, use_cftime=True)

        if len(remap_ds.data_vars) == 0:
            raise RuntimeError("Remapped file contains no data variables")

        delta_name = list(remap_ds.data_vars)[0]
        delta = remap_ds[delta_name]

        # CDO often leaves a dummy time axis of length 1
        if "time" in delta.dims and delta.sizes["time"] == 1:
            delta = delta.isel(time=0)

        # Make sure the result is a 2D lat/lon field
        if set(delta.dims) != {"lat", "lon"}:
            raise RuntimeError(
                f"Remapped field has unexpected dims {delta.dims}; expected ('lat', 'lon')"
            )

        # Verify grid match
        if delta.shape != (len(lat), len(lon)):
            raise RuntimeError(
                f"Remapped shape {delta.shape} does not match target grid {(len(lat), len(lon))}"
            )

        if not np.allclose(delta["lat"].values, lat):
            raise RuntimeError("Latitude coordinates do not match the target grid")

        if not np.allclose(delta["lon"].values, lon):
            raise RuntimeError("Longitude coordinates do not match the target grid")

        # Apply delta to the target variable. xarray will broadcast lat/lon
        # across batch/time automatically.
        delta_da = xr.DataArray(
            delta.values,
            coords={"lat": init_ds["lat"], "lon": init_ds["lon"]},
            dims=("lat", "lon"),
            name=f"{target_var}_delta",
        )

        out_ds = init_ds.copy()
        out_ds[target_var] = init_ds[target_var] + delta_da

        out_ds.to_netcdf(out_nc_path)

    return out_ds, remap_ds


out_ds, remap_ds = remap_grib_and_apply_to_sst(
    grib_path, init_nc_path, out_nc_path, target_var="sea_surface_temperature",
)