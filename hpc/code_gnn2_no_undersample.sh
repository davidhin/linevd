#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem 32G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4,singularity,matrix
#SBATCH --err="hpc/logs/code_gnn2_%j.info"
#SBATCH --output="hpc/logs/code_gnn2_%j.info"
#SBATCH --job-name="code_gnn"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

nvidia-smi

# Now need to retrain with the good dataset
   
singularity exec --nv main.sif python code_gnn/main.py --model flow_gnn --dataset MSR --feat "_ABS_DATAFLOW_datatypeonly" \
    --clean --batch_size 256 --max_epochs 250 --weight_decay 1e-2 --no_undersample_graphs \
    --label_style graph --cache_all \
    --evaluation
