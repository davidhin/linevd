#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem-per-cpu=8G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4,singularity,matrix
#SBATCH --err="hpc/logs/code_gnn_%A.info"
#SBATCH --output="hpc/logs/code_gnn_%A.info"
#SBATCH --job-name="code_gnn"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

nvidia-smi

# Now need to retrain with the good dataset

# 678722
# singularity exec --nv main.sif python code_gnn/main.py --model flow_gnn --dataset MSR \
#    --clean --batch_size 1024 --max_epochs 500 \
#    --label_style node --learning_rate 1e-3 --weight_decay 1e-2
   
# 678745
singularity exec --nv main.sif python code_gnn/main.py --model flow_gnn --dataset MSR \
   --clean --batch_size 1024 --max_epochs 500 \
   --label_style node --learning_rate 1e-3 --weight_decay 1e-2 --undersample_factor 1.0
