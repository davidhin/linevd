#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --array=1-100%5
#SBATCH --err="hpc/logs/getgraphs_sard_%a.err"
#SBATCH --output="hpc/logs/getgraphs_sard_%a.out"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name="getgraphs_sard"

# Setup Python Environment
module load singularity

# Start singularity instance
singularity exec main.sif python -u sastvd/scripts/getgraphs.py sard --job_array_number $SLURM_ARRAY_TASK_ID --workers 5
