#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-25%5
#SBATCH --output=absdf/logs/df1g-%A-%a.txt
#SBATCH --error=absdf/logs/df1g-%A-%a.txt
#SBATCH --job-name="df1g"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

/work/LAS/weile-lab/benjis/envs/linevd/bin/python -u sastvd/scripts/dataflow_joern.py --worker_id $SLURM_ARRAY_TASK_ID --n_splits 25
