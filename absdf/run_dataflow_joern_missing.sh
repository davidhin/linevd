#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 24G
#SBATCH --time=1-00:00:00
#SBATCH --array=11,12,16,18,20,21,23,24,2,3,4,7,8%1
#SBATCH --output=absdf/logs/df1g-%a.txt
#SBATCH --error=absdf/logs/df1g-%a.txt
#SBATCH --job-name="df1g_missing"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

/work/LAS/weile-lab/benjis/envs/linevd/bin/python -u sastvd/scripts/dataflow_joern.py --worker_id $SLURM_ARRAY_TASK_ID --n_splits 25
