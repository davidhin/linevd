#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --err="hpc/logs/sard_func.out"
#SBATCH --output="hpc/logs/sard_func.out"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name="sard_func"

# Setup Python Environment
module load singularity

# Start singularity instance
singularity exec main.sif pytest --disable-warnings -s sastvd/helpers/dataset_sard.py::test_sard_func