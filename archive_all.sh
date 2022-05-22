#!/bin/bash

#SBATCH --cpus-per-task 10
#SBATCH --time 1-00:00:00
#SBATCH --job-name="archive"
#SBATCH --output=repos/archive-%j.txt
#SBATCH --error=repos/archive-%j.txt
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

module load git
source activate.sh

python -c "import sastvd.scripts.get_repos as gr; gr.archive_commits()"
