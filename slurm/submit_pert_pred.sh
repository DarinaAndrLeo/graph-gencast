#!/bin/bash
#SBATCH --output=logs/submit_pert_pred-%j.out
#SBATCH --error=logs/submit_pert_pred-%j.err
#SBATCH --job-name=submit_pert_pred
set -euo pipefail

source ~/.bashrc
conda activate gencast

START=${1}
DAYS=${2}
ENSEMBLES=${3}
SEED=${4}
DELTA_KIND=${5}   # pi or incr

# GenCast uses the init time + 12h
START_SHORT="${START%:00:00}"

INIT_PATH="init_files/init_${START_SHORT}_${DAYS}d.nc"

# Output from the SST-delta step
if [[ "$DELTA_KIND" == "incr" ]]; then
    SST_OUT="init_files/1x1_${START_SHORT}_${DAYS}d_incrCO2_sstdelta.nc"
elif [[ "$DELTA_KIND" == "pi" ]]; then
    SST_OUT="init_files/1x1_${START_SHORT}_${DAYS}d_piCO2_sstdelta.nc"
else
    echo "DELTA_KIND must be 'pi' or 'incr'" >&2
    exit 1
fi

mkdir -p logs

prep_jobid=$(sbatch --parsable \
    --export=ALL,INIT_PATH="$INIT_PATH",SST_OUT="$SST_OUT",DELTA_KIND="$DELTA_KIND" \
    slurm/sst_delta_apply.sh)

echo "Submitted SST-delta job: $prep_jobid"

infer_jobid=$(sbatch --parsable \
    --dependency=afterok:"$prep_jobid" \
    slurm/run_gencast.sh "$START" "$DAYS" "$ENSEMBLES" "$SEED" "$SST_OUT" "$DELTA_KIND")

echo "Submitted inference job: $infer_jobid"
echo "Inference will start after SST-delta application finishes successfully."

#sbatch slurm/submit_pert_pred.sh 2021-06-25T12:00:00 10 26 0 pi