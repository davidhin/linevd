#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem 32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --exclude=amp-1,amp-2,amp-3,amp-4,singularity,matrix
#SBATCH --err="hpc/logs/code_gnn_%j.info"
#SBATCH --output="hpc/logs/code_gnn_%j.info"
#SBATCH --job-name="code_gnn"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

# Setup Python Environment
module load singularity gcc/7.3.0-xegsmw4 cuda/10.2.89-jveb27i

echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURM_JOB_NODELIST

nvidia-smi

feat="$1"
seed="$2"
if [ -z "$seed" ]
then
seed="0"
fi

shift
shift

echo "training $feat"
./mypython code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --clean --train_workers 4 --max_epochs 250 --weight_decay 1e-2 \
    --label_style graph --split random \
    --evaluation --seed $seed $@
