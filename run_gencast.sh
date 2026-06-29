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

START=${1}
DAYS=${2}
ENSEMBLES=${3}
SEED=${4}
START_SHORT=${START%:00:00}
START_PLUS12=$(python -c "
from datetime import datetime, timedelta
print((datetime.fromisoformat('$START') + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S'))
")
START_PLUS12_SHORT=${START_PLUS12%:00:00}

python run_gencast.py \
  --model "gencast_params_GenCast 1p0deg _2019.npz" \
  --data "init_files/init_${START_SHORT}_${DAYS}d.nc" \
  --output "/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/predictions/init_${START_PLUS12_SHORT}_${DAYS}d_${ENSEMBLES}ens_${SEED}seed_predictions.zarr" \
  --ensemble-size "$ENSEMBLES" \
  --seed "$SEED"

#submit with:
#sbatch run_gencast.sh 2021-06-17T12:00:00 30 8 0