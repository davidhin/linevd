#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4
#SBATCH --err="hpc/logs/code_gnn_%A.info"
#SBATCH --output="hpc/logs/code_gnn_%A.info"
#SBATCH --job-name="code_gnn"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

nvidia-smi

singularity exec --nv main.sif python code_gnn/main.py --model flow_gnn --dataset SARD \
    --clean --batch_size 1024 --max_epochs 1000 \
    --label_style node --undersample_factor 1.0
    #--node_type_separate
    #--undersample_factor 1.0
