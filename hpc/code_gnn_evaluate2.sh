#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem 32G
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4,singularity,matrix
#SBATCH --err="hpc/logs/code_gnn_eval2_%j.info"
#SBATCH --output="hpc/logs/code_gnn_eval2_%j.info"
#SBATCH --job-name="code_gnn_eval2"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

nvidia-smi

# Now need to retrain with the good dataset

for cp in logs/flow_gnn_MSR_graph__ABS_DATAFLOW_datatypeonly_None_None_None__0.001_0.01_256_5_2_32_False_0.5_sum_sum/default/checkpoints/periodical-*.ckpt
do
    echo Evaluating $cp...
    ls $cp || continue
    singularity exec --nv main.sif python code_gnn/main.py --model flow_gnn --dataset MSR --feat "_ABS_DATAFLOW_datatypeonly" \
        --clean --batch_size 256 --max_epochs 500 --weight_decay=1e-2 \
        --label_style graph --cache_all \
        --evaluation --resume_from_checkpoint $cp
done

