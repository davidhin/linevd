"""
Run get_dataflow_output.sc without caching, generating the following:
- .cpg.bin
- .nodes.json, .edges.json
- .dataflow.summary.json, .dataflow.*.json for each method
"""

import sastvd.helpers.datasets as svdds
import sastvd.helpers.joern_session as svdjs
import sastvd as svd
import tqdm


def get_dataflow_for_graph_split(worker_id, df):
    sess = svdjs.JoernSession(f"dataflow/{worker_id}", logfile=open(f"hpc/logs/dataflow_joern_{worker_id}.txt", "wb"))
    try:
        sess.import_script("get_dataflow_output")
        for fp in tqdm.tqdm(df["filepath"], desc="export dataflow"):
            sess.run_script("get_dataflow_output", params={"filename": fp, "cache": False}, import_first=False)
    finally:
        sess.close()

if __name__ == "__main__":
    import sys
    worker_id = int(sys.argv[1])
    n_splits = int(sys.argv[2])
    df = svdds.bigvul()
    print("original size", len(df))
    # TODO: correct the script to have id and filepath for before and after
    df["filepath"] = df["id"].apply(lambda _id: svd.processed_dir() / f"bigvul/after/{_id}.c")
    df = df[df["filepath"].apply(lambda fp: fp.exists())]
    print("filter to", len(df))
    test = sys.argv[3] == "test"
    if test:
        df = df.head(10)
    else:
        split_len = len(df) // n_splits
        df = df[worker_id*split_len:(worker_id+1)*split_len].copy()
        print(f"split", worker_id, df)
    print(df)
    get_dataflow_for_graph_split(worker_id, df)
