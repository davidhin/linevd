#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
#SBATCH --err="hpc/logs/code_gnn_cache_%j.info"
#SBATCH --output="hpc/logs/code_gnn_cache_%j.info"
#SBATCH --job-name="cgcache"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

feat=$1
seed=$2

echo "caching $feat"

singularity exec main.sif python -u code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --split random --dataset_only --skip_train --seed $seed
