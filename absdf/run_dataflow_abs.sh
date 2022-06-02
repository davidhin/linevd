#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --output=absdf/logs/dfabs-%j.txt
#SBATCH --error=absdf/logs/dfabs-%j.txt
#SBATCH --job-name="dfabs"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -u sastvd/scripts/abstract_dataflow.py --workers 16 --no-cache
