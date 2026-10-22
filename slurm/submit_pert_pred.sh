#!/bin/bash
#SBATCH --output=logs/submit_pert_pred-%j.out
#SBATCH --error=logs/submit_pert_pred-%j.err
#SBATCH --job-name=submit_pert_pred
set -euo pipefail

source ~/.bashrc
conda activate gencast

DATA_ROOT=${DATA_ROOT:-/network/group/aopp/predict/MK004_ANDRIYCHENKO_GENCAST/AI-IFS-attributioncomparison/pnw2021}

# INIT = forecast initialisation time = the LATEST input frame (the other input frame is init - 12h)
INIT=${1}          # e.g. 2021-06-26T00:00:00
DAYS=${2}          # forecast days after INIT
ENSEMBLES=${3}
SEED=${4}
DELTA_KIND=${5}    # pi or incr

INIT_SHORT="${INIT%:00:00}"

# Unperturbed (current-climate) init file that the SST delta is applied to
INIT_PATH="$DATA_ROOT/gencast/init/current/ic_1deg_current_init-${INIT_SHORT}_${DAYS}d.nc"

# Output from the SST-delta step
if [[ "$DELTA_KIND" == "incr" || "$DELTA_KIND" == "pi" ]]; then
    SST_OUT="$DATA_ROOT/gencast/init/${DELTA_KIND}/ic_1deg_${DELTA_KIND}_init-${INIT_SHORT}_${DAYS}d.nc"
else
    echo "DELTA_KIND must be 'pi' or 'incr'" >&2
    exit 1
fi

mkdir -p logs "$(dirname "$SST_OUT")"

prep_jobid=$(sbatch --parsable \
    --export=ALL,INIT_PATH="$INIT_PATH",SST_OUT="$SST_OUT",DELTA_KIND="$DELTA_KIND" \
    slurm/sst_delta_apply.sh)

echo "Submitted SST-delta job: $prep_jobid"

infer_jobid=$(sbatch --parsable \
    --dependency=afterok:"$prep_jobid" \
    slurm/run_gencast.sh "$INIT" "$DAYS" "$ENSEMBLES" "$SEED" "$SST_OUT" "$DELTA_KIND")

echo "Submitted inference job: $infer_jobid"
echo "Inference will start after SST-delta application finishes successfully."

#sbatch slurm/submit_pert_pred.sh 2021-06-26T00:00:00 10 26 0 pi
