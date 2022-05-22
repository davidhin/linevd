#!/bin/bash

#SBATCH --cpus-per-task 2
#SBATCH --mem 8G
#SBATCH --array=0-24%5
#SBATCH --output=repos/extract2-%A-%a.txt
#SBATCH --error=repos/extract2-%A-%a.txt
#SBATCH --time 3-00:00:00
#SBATCH --job-name="extract2"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -c "import sastvd.scripts.get_repos as gr; gr.extract_archived_commits($SLURM_ARRAY_TASK_ID, 25)"
