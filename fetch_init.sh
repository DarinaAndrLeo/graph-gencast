#!/bin/bash
#SBATCH --job-name=fetch_init
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source ~/.bashrc
conda activate gencast

START=${1}
DAYS=${2}
START_SHORT=${START%:00:00}
#start in format 2021-06-17T00:00:00

python fetch_init_weatherbench.py \
    --start "${START}" \
    --forecast-days "${DAYS}" \
    --output-file "init_files/init_${START_SHORT}_${DAYS}d.nc"

#submit with:
#sbatch fetch_init.sh 2021-06-25T00:00:00 30