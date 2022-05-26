#!/bin/bash

#SBATCH --cpus-per-task 2
#SBATCH --mem 32G

#SBATCH --array=0-99%5
#SBATCH --output=repos/parse3-%A-%a.txt
#SBATCH --error=repos/parse3-%A-%a.txt

#SBATCH --time 1-00:00:00
#SBATCH --job-name="parse3"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

source activate.sh

/work/LAS/weile-lab/benjis/envs/linevd/bin/python3.10 -c "import sastvd.scripts.get_repos as gr; gr.parse_with_joern($SLURM_ARRAY_TASK_ID, 100, 'repos/checkout3')"
# python -c "import sastvd.scripts.get_repos as gr; gr.parse_with_joern(-1)"
