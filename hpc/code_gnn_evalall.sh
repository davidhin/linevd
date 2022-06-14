#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem 16G
#SBATCH --time=1-00:00:00
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu
##SBATCH --exclude=amp-1,amp-2,amp-3,amp-4,singularity,matrix
#SBATCH --err="hpc/logs/code_gnn_evalall_%j.info"
#SBATCH --output="hpc/logs/code_gnn_evalall_%j.info"
#SBATCH --job-name="cgeva"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

nvidia-smi

feat="$1"
update_func="$2"
if [ -z "$update_func" ]
then
update_func="sum"
fi
seed="$3"
if [ -z "$seed" ]
then
seed="0"
fi

echo "training $feat"
./mypython code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --clean --batch_size 256 --train_workers 6 --max_epochs 250 --weight_decay 1e-2 \
    --label_style graph --split random \
    --evaluation --neighbor_pooling_type $update_func --seed $seed \
    --skip_train --resume_from_checkpoint "logs_5_truncated_untruncated/flow_gnn_MSR_graph_${feat}_None_None_undersample_None__0.001_0.01_256_${seed}_5_2_32_False_0.5_sum_sum/default" --take_checkpoint this
