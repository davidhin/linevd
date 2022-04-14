#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4
#SBATCH --err="hpc/logs/lvd_%A.info"
#SBATCH --output="hpc/logs/lvd_%A.info"
#SBATCH --job-name="lvd"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

# Start singularity instance
singularity exec --nv main.sif python -u sastvd/scripts/train_best.py
