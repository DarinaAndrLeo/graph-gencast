#!/bin/bash
#SBATCH --job-name=fetch_init
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
conda activate gencast

DATA_ROOT=${DATA_ROOT:-/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/AI-IFS-attributioncomparison/pnw2021}

# INIT = forecast initialisation time = the LATEST input frame (the other input frame is init - 12h)
INIT=${1}      # e.g. 2021-06-26T00:00:00
DAYS=${2}      # forecast days after INIT
INIT_SHORT=${INIT%:00:00}

OUT_DIR="$DATA_ROOT/gencast/init/current"
mkdir -p "$OUT_DIR"

python scripts/fetch_init_weatherbench.py \
    --init-time "${INIT}" \
    --forecast-days "${DAYS}" \
    --output-file "$OUT_DIR/ic_1deg_current_init-${INIT_SHORT}_${DAYS}d.nc"

#submit (from repo root) with:
#sbatch slurm/fetch_init.sh 2021-06-26T00:00:00 10
