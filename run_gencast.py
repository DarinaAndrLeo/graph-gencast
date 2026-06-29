#This is a file to run the gencast mini demo via slurm

# @title Imports

import dataclasses
import datetime
import haiku as hk
import jax
import numpy as np
import xarray
import argparse
import time

from graphcast import rollout
from graphcast import xarray_jax
from graphcast import normalization
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import xarray_tree
from graphcast import gencast
from graphcast import denoiser
from graphcast import nan_cleaning

start_time = time.time()

# @title Check example dataset matches model

def parse_file_parts(file_name):
  return dict(part.split("-", 1) for part in file_name.split("_"))

def data_valid_for_model(file_name: str, params_file_name: str):
  """Check data type and resolution matches."""
  data_file_parts = parse_file_parts(file_name.removesuffix(".nc"))
  res_matches = data_file_parts["res"].replace(".", "p") in params_file_name.lower()
  source_matches = "Operational" in params_file_name
  if data_file_parts["source"] == "era5":
    source_matches = not source_matches
  return res_matches and source_matches

# @title Build jitted functions, and possibly initialize random weights

def construct_wrapped_gencast():
  """Constructs and wraps the GenCast Predictor."""
  predictor = gencast.GenCast(
      sampler_config=sampler_config,
      task_config=task_config,
      denoiser_architecture_config=denoiser_architecture_config,
      noise_config=noise_config,
      noise_encoder_config=noise_encoder_config,
  )

  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level,
  )

  predictor = nan_cleaning.NaNCleaner(
      predictor=predictor,
      reintroduce_nans=True,
      fill_value=min_by_level,
      var_to_clean='sea_surface_temperature',
  )

  return predictor


@hk.transform_with_state
def run_forward(inputs, targets_template, forcings):
  predictor = construct_wrapped_gencast()
  return predictor(inputs, targets_template=targets_template, forcings=forcings)


@hk.transform_with_state
def loss_fn(inputs, targets, forcings):
  predictor = construct_wrapped_gencast()
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics),
  )


def grads_fn(params, state, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), i, t, f
    )
    return loss, (diagnostics, next_state)

  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True
  )(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads


# @title Set paths

#MODEL_PATH = "gencast_params_GenCast 1p0deg _2019.npz"  # E.g. "GenCast 1p0deg _2019.npz"
#DATA_PATH = "source-era5_date-2021-06-18_res-1.0_levels-13_steps-30.nc" #"source-era5_date-2019-03-29_res-1.0_levels-13_steps-30.nc"  # E.g. "source-era5_date-2019-03-29_res-1.0_levels-13_steps-04.nc"
#STATS_DIR = "stats/"  # E.g. "stats/"

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--ensemble-size", type=int, default=8,
                    help="Number of ensemble members")
parser.add_argument("--seed", type=int, required=True)#, default=0,
#                    help="Random seed")
args = parser.parse_args()

MODEL_PATH = args.model
DATA_PATH = args.data
OUT_PATH = args.output
STATS_DIR = "stats/"

# @title Load the model

with open(MODEL_PATH, "rb") as f:
  ckpt = checkpoint.load(f, gencast.CheckPoint)
params = ckpt.params
state = {}

task_config = ckpt.task_config
sampler_config = ckpt.sampler_config
noise_config = ckpt.noise_config
noise_encoder_config = ckpt.noise_encoder_config
denoiser_architecture_config = ckpt.denoiser_architecture_config
denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
denoiser_architecture_config.sparse_transformer_config.mask_type = "full"

#assert data_valid_for_model(DATA_PATH, MODEL_PATH)

# @title Load weather data

with open(DATA_PATH, "rb") as f:
  example_batch = xarray.load_dataset(f).compute()

assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets
print("Data loaded, start time:", example_batch.time.values[0], flush=True)

# @title Extract training and eval data

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", "12h"), # Only 1AR training.
    **dataclasses.asdict(task_config))

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times=slice("12h", f"{(example_batch.dims['time']-2)*12}h"), # All but 2 input frames.
    **dataclasses.asdict(task_config))

# @title Load normalization data

with open(STATS_DIR +"diffs_stddev_by_level.nc", "rb") as f:
  diffs_stddev_by_level = xarray.load_dataset(f).compute()
with open(STATS_DIR +"mean_by_level.nc", "rb") as f:
  mean_by_level = xarray.load_dataset(f).compute()
with open(STATS_DIR +"stddev_by_level.nc", "rb") as f:
  stddev_by_level = xarray.load_dataset(f).compute()
with open(STATS_DIR +"min_by_level.nc", "rb") as f:
  min_by_level = xarray.load_dataset(f).compute()


if params is None:
  init_jitted = jax.jit(loss_fn.init)
  params, state = init_jitted(
      rng=jax.random.PRNGKey(0),
      inputs=train_inputs,
      targets=train_targets,
      forcings=train_forcings,
  )


loss_fn_jitted = jax.jit(
    lambda rng, i, t, f: loss_fn.apply(params, state, rng, i, t, f)[0]
)
grads_fn_jitted = jax.jit(grads_fn)
run_forward_jitted = jax.jit(
    lambda rng, i, t, f: run_forward.apply(params, state, rng, i, t, f)[0]
)
# We also produce a pmapped version for running in parallel.
run_forward_pmap = xarray_jax.pmap(run_forward_jitted, dim="sample")

# @title Autoregressive rollout (loop in python)
num_ensemble_members = args.ensemble_size # @param int
rng = jax.random.PRNGKey(args.seed)
print(f"Ensemble size = {num_ensemble_members}, seed = {args.seed}", flush=True)
print("Starting rollout...", flush=True)
# We fold-in the ensemble member, this way the first N members should always
# match across different runs which use take the same inputs
# regardless of total ensemble size.
rngs = np.stack(
    [jax.random.fold_in(rng, i) for i in range(num_ensemble_members)], axis=0)

chunks = []
#for chunk in rollout.chunked_prediction_generator_multiple_runs(
    # Use pmapped version to parallelise across devices.
#    predictor_fn=run_forward_pmap,
#    rngs=rngs,
#    inputs=eval_inputs,
#    targets_template=eval_targets * np.nan,
#    forcings=eval_forcings,
#    num_steps_per_chunk = 1,
#    num_samples = num_ensemble_members,
#    pmap_devices=jax.local_devices()
#    ):
#    chunks.append(chunk)
for i, chunk in enumerate(
    rollout.chunked_prediction_generator_multiple_runs(
        predictor_fn=run_forward_pmap,
        rngs=rngs,
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        num_steps_per_chunk=1,
        num_samples=num_ensemble_members,
        pmap_devices=jax.local_devices()
    )
):
    chunks.append(chunk)
    print(f"Finished chunk {i+1}", flush=True)
predictions = xarray.combine_by_coords(chunks)

# This bit of code is to prepend the initialisation timesteps!
init_steps = example_batch.isel(time=slice(0, 2)).copy()
# Rewrite their time axis to match the forecast convention:
init_steps = init_steps.reset_coords("datetime", drop=True)
# step 1 = -12h, step 2 = 0h
init_steps = init_steps.assign_coords(
    time=np.array([-12, 0], dtype="timedelta64[h]")
)
# Prepend them to the forecast output.
predictions = xarray.concat([init_steps, predictions], dim="time")


#25mins ish?
elapsed = time.time() - start_time

print(
    f"Predictions obtained, saving... "
    f"(total runtime {elapsed/60:.1f} min)",
    flush=True,
)
predictions.to_zarr(OUT_PATH)