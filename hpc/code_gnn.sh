#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem 64G
#SBATCH --time=3-00:00:00
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
update_func="$2"
if [ -z "$2" ]
then
update_func="sum"
fi

echo "training $feat"

singularity exec --nv main.sif python -u code_gnn/main.py \
    --model flow_gnn --dataset MSR --feat $feat \
    --clean --batch_size 256 --max_epochs 500 --weight_decay 1e-2 \
    --label_style graph \
    --evaluation --neighbor_pooling_type $update_func
