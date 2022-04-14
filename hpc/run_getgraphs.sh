#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --time=48:00:00
#SBATCH --mem=48GB
#SBATCH --array=1-5
#SBATCH --err="hpc/logs/prepros_%a.err"
#SBATCH --output="hpc/logs/prepros_%a.out"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name="prepros"

# Setup Python Environment
module load singularity

# Start singularity instance
for i in $(seq 1 20)
do
  singularity exec main.sif python -u sastvd/scripts/getgraphs.py $(( $SLURM_ARRAY_TASK_ID * $i ))
done
