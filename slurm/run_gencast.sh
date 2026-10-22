#!/bin/bash
#SBATCH --job-name=gencast
#SBATCH --partition=gputest
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/.bashrc
conda activate gencast

DATA_ROOT=${DATA_ROOT:-/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/AI-IFS-attributioncomparison/pnw2021}

# INIT = forecast initialisation time = the LATEST input frame (the other input frame is init - 12h)
INIT=${1}          # e.g. 2021-06-26T00:00:00
DAYS=${2}          # forecast days after INIT
ENSEMBLES=${3}
SEED=${4}
DATA=${5:-}        # optional: explicit init file (used for perturbed runs)
DELTA_KIND=${6:-}  # optional: pi | incr (empty = current climate)

INIT_SHORT=${INIT%:00:00}
STATE=${DELTA_KIND:-current}

if [[ -n "$DATA" ]]; then
    DATA_PATH="$DATA"
else
    DATA_PATH="$DATA_ROOT/gencast/init/current/ic_1deg_current_init-${INIT_SHORT}_${DAYS}d.nc"
fi

OUT_DIR="$DATA_ROOT/gencast/predictions/1deg/$STATE/init-${INIT_SHORT}_${DAYS}d"
mkdir -p "$OUT_DIR"

python scripts/run_gencast.py \
  --model "data/params/gencast_params_GenCast 1p0deg _2019.npz" \
  --data "$DATA_PATH" \
  --output "$OUT_DIR/seed${SEED}_${ENSEMBLES}ens.zarr" \
  --ensemble-size "$ENSEMBLES" \
  --seed "$SEED"

#submit (from repo root) with:
#sbatch slurm/run_gencast.sh 2021-06-26T00:00:00 10 26 0
