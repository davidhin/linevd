#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --err="hpc/logs/first_%A.info"
#SBATCH --output="hpc/logs/first_%A.info"
#SBATCH --job-name="first"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python sastvd/linevd/plot_first_rates.py
