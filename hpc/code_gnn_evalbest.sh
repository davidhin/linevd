#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem 64G
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4,singularity,matrix
#SBATCH --err="hpc/logs/code_gnn_eval_%j.info"
#SBATCH --output="hpc/logs/code_gnn_eval_%j.info"
#SBATCH --job-name="cgeval"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

nvidia-smi

feat=$1

echo "training $feat"

singularity exec --nv main.sif python -u code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --batch_size 256 --weight_decay 1e-2 \
    --label_style graph --cache_all \
    --evaluation --skip_train --log_suffix _eval --take_checkpoint best
