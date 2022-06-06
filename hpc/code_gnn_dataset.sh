#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 8G
#SBATCH --time=1-00:00:00
#SBATCH --err="hpc/logs/code_gnn_cache_%j.info"
#SBATCH --output="hpc/logs/code_gnn_cache_%j.info"
#SBATCH --job-name="cgcache"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

feat=$1

echo "caching $feat"

/work/LAS/weile-lab/benjis/envs/linevd/bin/python code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --batch_size 256 \
    --label_style graph \
    --evaluation --dataset_only
