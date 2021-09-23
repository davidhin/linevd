#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=48:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --err="hpc/logs/mo_%A.info"
#SBATCH --output="hpc/logs/mo_%A.info"
#SBATCH --job-name="mo"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python -u sastvd/scripts/run_method.py
