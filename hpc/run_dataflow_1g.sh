#!/bin/bash
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem 32G
#SBATCH --time=1-00:00:00
#SBATCH --output=hpc/logs/df1g-%j.txt
#SBATCH --error=hpc/logs/df1g-%j.txt
#SBATCH --job-name="df1g"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -u sastvd/scripts/dataflow_1g.py
