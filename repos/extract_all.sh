#!/bin/bash

#SBATCH --cpus-per-task 1
#SBATCH --mem 16G
#SBATCH --array=0-2%5
#SBATCH --output=repos/extract3-%A-%a.txt
#SBATCH --error=repos/extract3-%A-%a.txt
#SBATCH --time 3-00:00:00
#SBATCH --job-name="extract3"
#SBATCH --mail-user=benjis@iastate.edu
#SBATCH --mail-type=FAIL,END

fn=`printf "archives_to_extract3_%02d" $SLURM_ARRAY_TASK_ID`
wc -l $fn
while read l
do
    echo extracting $l
    bl=repos/checkout3/`basename ${l%.tar}`
    mkdir -p $bl
    tar xf "$l" -C "$bl"
done < $fn
