#!/bin/bash
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --job-name=gencast
set -euo pipefail

START=${1}
DAYS=${2}
ENSEMBLES=${3}

START_PLUS12=$(date -d "$START +12 hours" '+%Y-%m-%dT%H:%M:%S')
START_PLUS12_SHORT=${START_PLUS12%:00:00}
DATA="init_files/init_${START_PLUS12_SHORT}_${DAYS}d.nc"

prep_jobid=$(sbatch --parsable fetch_init.sh "$START" "$DAYS")
echo "Submitted prep job: $prep_jobid"

infer_jobid=$(sbatch --parsable --dependency=afterok:"$prep_jobid" run_gencast.sh "$START" "$DAYS" "$ENSEMBLES" "$SEED")
echo "Submitted inference job: $infer_jobid"
echo "Inference will start after prep finishes successfully."

#sbatch submit_init_pred.sh 2021-06-18T00:00:00 30 8