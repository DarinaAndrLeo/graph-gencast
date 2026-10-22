#!/bin/bash
#SBATCH --output=logs/sst_delta-%j.out
#SBATCH --error=logs/sst_delta-%j.err
#SBATCH --job-name=sst_delta
#SBATCH --cpus-per-task=1

set -euo pipefail

source ~/.bashrc
conda activate gencast

# Allow either:
# 1) environment variables from sbatch --export, or
# 2) direct command-line use:
# ./slurm/sst_delta_apply.sh INIT_PATH SST_OUT [pi|incr]

INIT_PATH="${INIT_PATH:-${1:-}}"
SST_OUT="${SST_OUT:-${2:-}}"
DELTA_KIND="${DELTA_KIND:-${3:-pi}}"

: "${INIT_PATH:?INIT_PATH must be set or passed as the first argument}"
: "${SST_OUT:?SST_OUT must be set or passed as the second argument}"

if [[ "$DELTA_KIND" == "incr" ]]; then
PI_OR_INCR=1
elif [[ "$DELTA_KIND" == "pi" ]]; then
PI_OR_INCR=-1
else
echo "DELTA_KIND must be 'pi' or 'incr'" >&2
exit 1
fi

python scripts/sst_delta_apply.py \
    --init-file "$INIT_PATH" \
    --pi-or-incr "$PI_OR_INCR" \
    --output-file "$SST_OUT"

#./slurm/sst_delta_apply.sh init_files/init_2021-06-18T12:00:00_30d.nc init_files/init_2021-06-18T12:00:00_30d_piCO2_sstdelta.nc pi