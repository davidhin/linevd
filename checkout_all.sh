#!/bin/bash

#SBATCH --array=0-9%5
#SBATCH --output=repos/checkout-%A-%a.txt
#SBATCH --error=repos/checkout-%A-%a.txt
#SBATCH --time 1-00:00:00 
#SBATCH --job-name="checkout"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

module load git
source activate.sh

python -c "import sastvd.scripts.get_repos as gr; gr.get_repos_commits($SLURM_ARRAY_TASK_ID)"
