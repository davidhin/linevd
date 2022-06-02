#!/bin/bash
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --err="hpc/logs/prepare.out"
#SBATCH --output="hpc/logs/prepare.out"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name="prepare"

source activate.sh

# Start singularity instance
python -u sastvd/scripts/prepare.py --global_workers 12
