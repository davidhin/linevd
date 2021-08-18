#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --err="hpc/logs/prepare.out"
#SBATCH --output="hpc/logs/prepare.out"
#SBATCH --job-name="prepare"

# Setup Python Environment
module load Singularity

# Start singularity instance
singularity exec -H /g/acvt/a1720858/sastvd main.sif python -u sastvd/scripts/prepare.py