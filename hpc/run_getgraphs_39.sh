#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=1-00:00:00
#SBATCH --mem=48GB
#SBATCH --array=38-39%5
#SBATCH --err="hpc/logs/getgraphs3839_%a.out"
#SBATCH --output="hpc/logs/getgraphs3839_%a.out"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name="getgraphs3839"

source activate.sh

# Start singularity instance
python -u sastvd/scripts/getgraphs.py bigvul --sess --job_array_number $SLURM_ARRAY_TASK_ID --num_jobs 100
