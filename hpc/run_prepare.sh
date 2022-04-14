#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --err="hpc/logs/prepare.out"
#SBATCH --output="hpc/logs/prepare.out"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name="prepare"

# Setup Python Environment
module load singularity

# Start singularity instance
singularity exec main.sif python -u sastvd/scripts/prepare.py
