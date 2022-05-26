#!/bin/bash

#SBATCH --output=repos/dl-%j.txt
#SBATCH --error=repos/dl-%j.txt
#SBATCH --time 1-00:00:00 
#SBATCH --job-name="dl"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

module load git

mkdir -p repos

while read l
do
    git clone $l repos/$l
done < codeLinks.txt
