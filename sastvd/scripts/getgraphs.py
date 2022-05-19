import functools
from multiprocessing import Pool
import os
import pickle as pkl
import pexpect
import traceback

import numpy as np
import tqdm
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.joern_session as svdjs
import sastvd.helpers.sast as sast


def preprocess(row, fn):
    """Parallelise svdj functions.

    Example:
    df = svdd.bigvul()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    try:

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
                fn(filepath=fpath, verbose=args.verbose)
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
            if args.test or not (os.path.exists(f"{fpath1}.edges.json") or os.path.exists(f"{fpath1}.cpg.bin")):
                fn(filepath=fpath1, verbose=args.verbose)

            # Run Joern on "after" code
            if len(row["diff"]) > 0 and (args.test or not (os.path.exists(f"{fpath2}.edges.json") or os.path.exists(f"{fpath2}.cpg.bin"))):
                fn(filepath=fpath2, verbose=args.verbose)

            # Run SAST extraction
            if args.run_sast:
                fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
                if not os.path.exists(fpath3):
                    sast_before = sast.run_sast(row["before"])
                    with open(fpath3, "wb") as f:
                        pkl.dump(sast_before, f)
    except Exception:
        with open("failed_joern.txt", "a") as f:
            print(f"ERROR {row['id']}: {traceback.format_exc()}\ndata={row}", file=f)


def test_preprocess():
    """
    test that preprocessing progresses alright
    """
    row = {}
    result = preprocess(row)
    print(f"{result}")


def preprocess_whole_df_split(t):
    """
    preprocess one split of the dataframe
    """
    i, split = t
    sess = svdjs.JoernSession(i)
    try:
        fn = functools.partial(svdj.run_joern_sess, sess=sess, export_json=False, export_cpg=True)
        items = split.to_dict("records")
        for row in tqdm.tqdm(items, desc=f"(worker {i})", position=i):
            preprocess(row, fn)
    finally:
        sess.close()


def preprocess_whole_df(df):
    """
    preprocess entire dataframe with 1 split in each thread
    """
    splits = np.array_split(df, args.workers)
    with Pool(processes=args.workers) as p:
        for _ in tqdm.tqdm(p.imap_unordered(preprocess_whole_df_split, enumerate(splits, start=1)), desc=f"overall progress ({args.workers} workers)", position=0, total=len(splits)):
            pass
        # NOTE: single threaded run for cProfile
        # for p in enumerate(splits, start=1):
        #     preprocess_whole_df_split(p)


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
        df = svdd.bigvul(sample=args.test)
    if args.dataset == "sard":
        df = svds.sard()

    # Read Data
    df = df.reset_index().iloc[::-1]
    if args.test:
        print("test - sampling 10 examples")
        df = df.sample(10)
        args.verbose = 4
    
    if args.job_array_number is not None:
        splits = np.array_split(df, args.num_jobs)
        split_number = int(args.job_array_number) - 1
        df = splits[split_number]
        svd.dfmp(df, functools.partial(preprocess, fn=svdj.run_joern), ordr=False, workers=args.workers)
    else:
        preprocess_whole_df(df)
