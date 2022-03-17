# LineVD

This repository provides the code for [LineVD: Statement-level Vulnerability Detection using Graph Neural Networks](https://arxiv.org/pdf/2203.05181.pdf). The environment can be built using [Singularity](https://sylabs.io/singularity/), or by following / following the commands in the Singularity file. To start, clone the repository and navigate to the root directory.

## Directory Structure

```dir
(main module) ├── sastvd
              │   ├── codebert
              │   ├── helpers
              │   ├── ivdetect
              │   ├── linevd
              │   └── scripts
              ├── storage
(memoization) │   ├── cache
(raw data)    │   ├── external
(csvs)        │   ├── outputs
(models)      │   └── processed
(tests)       └── tests
```

## Training LineVD from scratch

Build and initialise environment and download dataset

```sh
sudo singularity build main.sif Singularity
singularity run main.sif -p initialise
```

Feature extraction (Increase NUM_JOBS if running on HPC for speed up)

```sh
singularity exec main.sif python sastvd/scripts/prepare.py
singularity exec main.sif python sastvd/scripts/getgraphs.py
```

Train model (Training takes around 1-2 hours using GTX 3060)

```sh
singularity exec --nv main.sif python sastvd/scripts/train_best.py
```
