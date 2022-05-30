#!/bin/bash

#SBATCH --cpus-per-task 2
#SBATCH --mem 64G

#SBATCH --array=0-99%5
#SBATCH --output=repos/logs/get_abs_df-%A-%a.txt
#SBATCH --error=repos/logs/get_abs_df-%A-%a.txt

#SBATCH --time 1-00:00:00
#SBATCH --job-name="get_abs_df"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

/work/LAS/weile-lab/benjis/envs/linevd/bin/python3.10 sastvd/scripts/get_abs_df.py $SLURM_ARRAY_TASK_ID 100 no
