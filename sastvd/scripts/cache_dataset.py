import sastvd.linevd as lvd
import functools
import multiprocessing
import tqdm
import numpy as np
from code_gnn.main import visualize_example
from pathlib import Path
import traceback

test = False

storage_dir = Path("storage/processed/bigvul")

def do_one(ds, max_df_dim, split, dataargs, t):
    i, ids = t
    for _id in tqdm.tqdm(ids, desc="cache_items", position=i):
        try:
            g = ds.item(_id, max_dataflow_dim=max_df_dim)
            # b_fpath = storage_dir / "before" / (str(_id) + ".c")
            # b = b_fpath.open().read() if b_fpath.exists() else None
            # a_fpath = storage_dir / "after" / (str(_id) + ".c")
            # a = a_fpath.open().read() if a_fpath.exists() else None
            # visualize_example(g, b, a, _id=_id)
        except Exception as E:
            print("exception", traceback.format_exc())
        if test:
            print("got graph", g, g.ndata["_DATAFLOW"].sum().item())

if __name__ == "__main__":

    dataargs = {
        "sample": 100 if test else -1,
        "gtype": "cfg",
        "splits": "default",
        "feat": "all"
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
    nproc = 12
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