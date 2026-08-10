import numpy as np
import xarray as xr

p1 = "/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/predictions/init_2021-06-18T00_26d_26ens_0seed_predictions.zarr"
p2 = "/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/predictions/init_2021-06-18T00_26d_24ens_1seed_predictions.zarr"
out = "/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/predictions/init_2021-06-18T00_26d_50ens_26-0s_24-1s_predictions.zarr"

ds1 = xr.open_zarr(p1)
ds2 = xr.open_zarr(p2)

# Give the second file unique sample labels so you end up with 0..49
ds2 = ds2.assign_coords(
    sample=np.arange(ds1.sizes["sample"], ds1.sizes["sample"] + ds2.sizes["sample"])
)

combined = xr.concat(
    [ds1, ds2],
    dim="sample",
    data_vars="minimal",
    coords="minimal",
    compat="override",
    combine_attrs="override",
)

combined.to_zarr(
    out,
    mode="w",
    consolidated=True,
    align_chunks=True,
)
print(combined)
print(f"Wrote combined Zarr to: {out}")