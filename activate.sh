module load openjdk git
source /work/LAS/weile-lab/benjis/anaconda3/etc/profile.d/conda.sh
conda activate linevd
export PYTHONPATH=$PWD
export PATH="$PATH:$PWD/storage/external/joern/joern-cli"
