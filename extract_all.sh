#!/bin/bash

#SBATCH --cpus-per-task 25
#SBATCH --array=0-99%5
#SBATCH --output=repos/extract-%A-%a.txt
#SBATCH --error=repos/extract-%A-%a.txt
#SBATCH --time 3-00:00:00
#SBATCH --job-name="extract"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -c "import sastvd.scripts.get_repos as gr; gr.extract_archived_commits($SLURM_ARRAY_TASK_ID, 100)"
