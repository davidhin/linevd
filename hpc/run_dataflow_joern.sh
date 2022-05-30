#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 8G
#SBATCH --time=1-00:00:00
# #SBATCH --array=0-100%5
#SBATCH --array=0-20%5
#SBATCH --output=hpc/logs/dataflow-%A-%a.txt
#SBATCH --error=hpc/logs/dataflow-%A-%a.txt
#SBATCH --job-name="dataflow_1g"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

# Running 187072 samples in /before
# /work/LAS/weile-lab/benjis/envs/linevd/bin/python -u sastvd/scripts/dataflow_joern.py $SLURM_ARRAY_TASK_ID 100 no

# Running 9336 samples in /after
/work/LAS/weile-lab/benjis/envs/linevd/bin/python -u sastvd/scripts/dataflow_joern.py $SLURM_ARRAY_TASK_ID 20 no
