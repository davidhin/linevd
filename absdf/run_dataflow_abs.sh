#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --output=absdf/logs/dfabs.txt
#SBATCH --error=absdf/logs/dfabs.txt
#SBATCH --job-name="dfabs"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -u sastvd/scripts/abstract_dataflow_full.py --workers 16 --no-cache --stage 1
