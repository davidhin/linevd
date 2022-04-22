#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 8G
#SBATCH --time=2-00:00:00
#SBATCH --err="hpc/logs/cache_dataset_%j.info"
#SBATCH --output="hpc/logs/cache_dataset_%j.info"
#SBATCH --job-name="cache dataset"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

# Start singularity instance
singularity exec main.sif python -u sastvd/scripts/cache_dataset.py
