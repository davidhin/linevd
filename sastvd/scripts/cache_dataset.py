import functools
import multiprocessing
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import sastvd.linevd as lvd
import tqdm
from code_gnn.main import visualize_example

test = False

storage_dir = Path("storage/processed/bigvul")

def do_one(ds, max_df_dim, split, dataargs, t):
    i, ids = t
    # for _id in tqdm.tqdm(ids, desc="cache_items", position=i):
    log_every = len(ids) // 10
    for j, _id in enumerate(ids):
        try:
            if j != 0 and j % log_every == 0:
                print("do_one", i, j, datetime.now())
            g = ds.item(_id, max_dataflow_dim=max_df_dim)
        except Exception as E:
            print("exception", traceback.format_exc())
        if test:
            print("got graph", g, g.ndata["_DATAFLOW"].sum().item())

if __name__ == "__main__":

    dataargs = {
        "sample": 100 if test else -1,
        "gtype": "cfg",
        "splits": "default",
        "feat": "_ABS_DATAFLOW"
    }

    if test:
        max_df_dim = 223*2  # DEBUG: hardcoded
    else:
        max_df_dim = 0
        for split in ["train", "val", "test"]:
            max_df_dim = lvd.BigVulDatasetLineVD(partition=split, **dataargs).get_max_dataflow_dim(max_df_dim)
            print("max_df_dim", max_df_dim)

    dataargs["max_df_dim"] = max_df_dim

    # Load all data
    # breakpoint()
    # nproc = 3 if test else 12
    nproc = 1 if test else 12
    with multiprocessing.Pool(nproc) as pool:
        for split in ["train", "val", "test"]:
            ds = lvd.BigVulDatasetLineVD(partition=split, **dataargs)
            ids = ds.df.sample(len(ds.df)).id.tolist()
            fn = functools.partial(do_one, ds, max_df_dim, split, dataargs)

            del ds.df
            splits = np.array_split(ids, nproc)
            for _ in pool.imap_unordered(fn, enumerate(splits)):
                pass
            
            # do_one((0, ids))
