import os
import pickle as pkl
import sys

import numpy as np
import sastvd as svd
import sastvd.helpers.dataset_sard as svds
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast


def preprocess(row, run_sast=True):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """

    if row["dataset"] == "sard":
        savedir = svd.get_dir(svd.processed_dir() / row["dataset"] / "code")
        # print(row)

        # Write C Files
        fpath = savedir / f"{row['index']}.c"
        if not fpath.exists():
            with open(fpath, "w") as f:
                f.write(row["code"])

        # Run Joern on code
        if not os.path.exists(f"{fpath}.edges.json") or args.test:
            svdj.full_run_joern(fpath, verbose=args.verbose)
    if row["dataset"] == "bigvul":
        savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
        savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")

        # Write C Files
        fpath1 = savedir_before / f"{row['id']}.c"
        with open(fpath1, "w") as f:
            f.write(row["before"])
        fpath2 = savedir_after / f"{row['id']}.c"
        if len(row["diff"]) > 0:
            with open(fpath2, "w") as f:
                f.write(row["after"])

        # Run Joern on "before" code
        if not os.path.exists(f"{fpath1}.edges.json") or args.test:
            svdj.full_run_joern(fpath1, verbose=args.verbose)

        # Run Joern on "after" code
        if (not os.path.exists(f"{fpath2}.edges.json") or args.test) and len(row["diff"]) > 0:
            svdj.full_run_joern(fpath2, verbose=args.verbose)

        # Run SAST extraction
        if args.run_sast:
            fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
            if not os.path.exists(fpath3):
                sast_before = sast.run_sast(row["before"])
                with open(fpath3, "wb") as f:
                    pkl.dump(sast_before, f)


def test_preprocess():
    row = {}
    result = preprocess(row)
    print(f"{result}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["bigvul", "sard"])
    parser.add_argument("--job_array_number")
    parser.add_argument("--num_jobs", default=100, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--run_sast", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--verbose", type=int, default=3)
    args = parser.parse_args()

    if args.dataset == "bigvul":
        df = svdd.bigvul()
    if args.dataset == "sard":
        df = svds.sard()

    # SETUP
    NUM_JOBS = args.num_jobs
    JOB_ARRAY_NUMBER = 0 if "ipykernel" in sys.argv[0] else int(args.job_array_number) - 1

    # Read Data
    df = df.reset_index().iloc[::-1]
    if args.test:
        print("TEST - getting head() only")
        df = df.head()
        args.verbose = 4
    else:
        splits = np.array_split(df, NUM_JOBS)
        df = splits[JOB_ARRAY_NUMBER]
    svd.dfmp(df, preprocess, ordr=False, workers=args.workers)
