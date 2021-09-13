#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=48:00:00
#SBATCH --mem=48GB
#SBATCH --array=1-100
#SBATCH --err="hpc/logs/prepros_%a.err"
#SBATCH --output="hpc/logs/prepros_%a.out"
#SBATCH --job-name="prepros"

# Setup Python Environment
module load Singularity

# Start singularity instance
singularity exec -H /g/acvt/a1720858/sastvd main.sif python -u sastvd/scripts/getgraphs.py $SLURM_ARRAY_TASK_ID
