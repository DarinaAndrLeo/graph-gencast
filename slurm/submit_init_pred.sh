#!/bin/bash
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --job-name=gencast
set -euo pipefail

# INIT = forecast initialisation time = the LATEST input frame (the other input frame is init - 12h)
INIT=${1}          # e.g. 2021-06-26T00:00:00
DAYS=${2}          # forecast days after INIT
ENSEMBLES=${3}
SEED=${4}

prep_jobid=$(sbatch --parsable slurm/fetch_init.sh "$INIT" "$DAYS")
echo "Submitted prep job: $prep_jobid"

infer_jobid=$(sbatch --parsable --dependency=afterok:"$prep_jobid" slurm/run_gencast.sh "$INIT" "$DAYS" "$ENSEMBLES" "$SEED")
echo "Submitted inference job: $infer_jobid"
echo "Inference will start after prep finishes successfully."

#sbatch slurm/submit_init_pred.sh 2021-06-26T00:00:00 10 26 0
