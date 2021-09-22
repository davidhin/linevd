#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --err="hpc/logs/empan_%A.info"
#SBATCH --output="hpc/logs/empan_%A.info"
#SBATCH --job-name="empan"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python sastvd/linevd/empirical_eval.py
