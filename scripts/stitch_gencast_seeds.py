"""Stitch ensemble runs made with different seeds into one store along `sample`.

Looks in <data-root>/gencast/predictions/1deg/<state>/init-<init>_<N>d/ for the stores written by
slurm/run_gencast.sh (pred_1deg_<state>_init-<init>_<N>d_seed<S>_<M>ens.zarr), one per seed, and
writes pred_1deg_<state>_init-<init>_<N>d_stitched_<total>ens.zarr next to them.

Example (26 members from seed 0 + 24 from seed 1 -> 50):
    python scripts/stitch_gencast_seeds.py --init-time 2021-06-26T00:00:00 --days 10 --state pi --seeds 0 1
"""
import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

DEFAULT_ROOT = "/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/AI-IFS-attributioncomparison/pnw2021"

parser = argparse.ArgumentParser()
parser.add_argument("--init-time", required=True, help="Forecast initialisation time (latest input frame), e.g. 2021-06-26T00:00:00")
parser.add_argument("--days", required=True, type=int, help="Forecast days after init, as in the file names")
parser.add_argument("--state", required=True, choices=["current", "pi", "incr"])
parser.add_argument("--seeds", required=True, type=int, nargs="+", help="Seeds to stitch, in the order they are concatenated, e.g. 0 1")
parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", DEFAULT_ROOT))
parser.add_argument("--overwrite", action="store_true", help="Replace the stitched store if it already exists")
args = parser.parse_args()

init_short = datetime.fromisoformat(args.init_time).strftime("%Y-%m-%dT%H")
tag = f"pred_1deg_{args.state}_init-{init_short}_{args.days}d"
run_dir = Path(args.data_root) / "gencast" / "predictions" / "1deg" / args.state / f"init-{init_short}_{args.days}d"

datasets = []
offset = 0
for seed in args.seeds:
    matches = sorted(run_dir.glob(f"{tag}_seed{seed}_*ens.zarr"))
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one store for seed {seed} in {run_dir}, found {len(matches)}")
    ds = xr.open_zarr(matches[0])
    # Give each seed's members unique sample labels so you end up with 0..N-1
    ds = ds.assign_coords(sample=np.arange(offset, offset + ds.sizes["sample"]))
    offset += ds.sizes["sample"]
    print(f"seed {seed}: {matches[0].name} ({ds.sizes['sample']} members)")
    datasets.append(ds)

out = run_dir / f"{tag}_stitched_{offset}ens.zarr"
if out.exists():
    if not args.overwrite:
        raise SystemExit(f"{out} already exists (use --overwrite to replace it)")
    shutil.rmtree(out)

combined = xr.concat(
    datasets,
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
