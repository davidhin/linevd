#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --err="hpc/logs/vis_%A.info"
#SBATCH --output="hpc/logs/vis_%A.info"
#SBATCH --job-name="vis"

# Setup Python Environment
module load Singularity
module load CUDA/10.2.89

# Start singularity instance
singularity exec -H /g/acvt/a1720858/sastvd --nv main.sif python sastvd/linevd/generate_pred_vis.py
