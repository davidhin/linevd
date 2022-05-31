#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem 8G
#SBATCH --time=1-00:00:00
#SBATCH --output=hpc/logs/df1g-%j.txt
#SBATCH --error=hpc/logs/df1g-%j.txt
#SBATCH --job-name="dataflow_1g"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python sastvd/scripts/dataflow_1g.py
