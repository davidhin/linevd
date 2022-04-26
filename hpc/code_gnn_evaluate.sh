#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
# SBATCH --gres=gpu:1
# SBATCH --partition=gpu
# SBATCH --exclude=amp-1,amp-2,amp-3,amp-4
#SBATCH --err="hpc/logs/code_gnn_eval_%A.info"
#SBATCH --output="hpc/logs/code_gnn_eval_%A.info"
#SBATCH --job-name="code_gnn_eval"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 #cuda/10.2.89-jveb27i

# nvidia-smi

# singularity exec --nv main.sif $@
# cpdir="logs/flow_gnn_SARD_None_None_None__0.001_0_1024_5_2_32_False_0.5_sum_sum/default/checkpoints"
# for cp in periodical-100-1414.ckpt periodical-250-3514.ckpt periodical-500-7014.ckpt
# do
# echo "Evaluating $cpdir/$cp..."
# singularity exec main.sif python -u code_gnn/main.py --model flow_gnn --dataset SARD --batch_size 1024 --label_style node \
#     --evaluation --resume_from_checkpoint "$cpdir/$cp"
# done

# cpdir="logs/flow_gnn_SARD_None_None_1.0__0.001_0_1024_5_2_32_False_0.5_sum_sum/default/checkpoints"
# for cp in periodical-100-1414.ckpt periodical-250-3514.ckpt periodical-500-7014.ckpt
# do
# echo "Evaluating $cpdir/$cp..."
# singularity exec main.sif python -u code_gnn/main.py --model flow_gnn --dataset SARD --batch_size 1024 --label_style node --undersample_factor 1.0 \
#     --evaluation --resume_from_checkpoint "$cpdir/$cp"
# done


# 2nd run... with weight decay. Got cut short.
cpdir="logs/flow_gnn_SARD_None_None_1.0__0.001_0.01_1024_5_2_32_False_0.5_sum_sum/default/checkpoints"
for cp in periodical-100-1414.ckpt periodical-250-3514.ckpt #periodical-500-7014.ckpt
do
echo "Evaluating $cpdir/$cp..."
singularity exec main.sif python -u code_gnn/main.py --model flow_gnn --dataset SARD --batch_size 1024 --label_style node --undersample_factor 1.0 --weight_decay 1e-2 \
    --evaluation --resume_from_checkpoint "$cpdir/$cp"
done

cpdir="logs/flow_gnn_SARD_None_None_1.0__0.0001_0.01_1024_5_2_32_False_0.5_sum_sum/default/checkpoints"
for cp in periodical-100-1414.ckpt periodical-250-3514.ckpt #periodical-500-7014.ckpt
do
echo "Evaluating $cpdir/$cp..."
singularity exec main.sif python -u code_gnn/main.py --model flow_gnn --dataset SARD --batch_size 1024 --label_style node --undersample_factor 1.0 --learning_rate 1e-4 --weight_decay 1e-2 \
    --evaluation --resume_from_checkpoint "$cpdir/$cp"
done