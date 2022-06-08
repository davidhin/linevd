#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --output=absdf/logs/dfabs2.txt
#SBATCH --error=absdf/logs/dfabs2.txt
#SBATCH --job-name="dfabs2"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -u sastvd/scripts/abstract_dataflow_full.py --workers 16 --stage 2 --cache
