import code_gnn.analysis.dataflow as cgdf
import sastvd.helpers.datasets as svdds
import sastvd.helpers.joern_session as svdjs
import sastvd as svd

import tqdm
import functools
import pandas as pd
import traceback
import json
from multiprocessing import Pool

def get_dataflow_for_graph_split(worker_id, graph_ids):
    sess = svdjs.JoernSession(f"dataflow/{worker_id}", logfile=open(f"output_dataflow_joern_{worker_id}.txt", "wb"))
    try:
        sess.import_script("get_dataflow_output")
        for _id in graph_ids:
            sess.run_script("get_dataflow_output", params={"filename": svdds.itempath(_id), "cache": False}, import_first=False)
    finally:
        sess.close()

if __name__ == "__main__":
    import sys
    worker_id = int(sys.argv[1])
    n_splits = int(sys.argv[2])
    df = svdds.bigvul()
    test = sys.argv[3] == "test"
    if test:
        df = df.head(100)
    print(df)
    split_len = len(df) // n_splits
    df = df[worker_id*split_len:(worker_id+1)*split_len].copy()
    print(f"split", worker_id, df)
    get_dataflow_for_graph_split(worker_id, df["id"])
