#!/bin/bash

#SBATCH --cpus-per-task 3
#SBATCH --mem 16G

#SBATCH --array=0-99%5
#SBATCH --output=repos/parse-%A-%a.txt
#SBATCH --error=repos/parse-%A-%a.txt

##SBATCH --output=repos/parse-%j.txt
##SBATCH --error=repos/parse-%j.txt

#SBATCH --time 1-00:00:00
#SBATCH --job-name="parse"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

python -c "import sastvd.scripts.get_repos as gr; gr.parse_with_joern($SLURM_ARRAY_TASK_ID, 100)"
# python -c "import sastvd.scripts.get_repos as gr; gr.parse_with_joern(-1)"
