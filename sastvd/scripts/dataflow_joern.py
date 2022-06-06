"""
Run get_dataflow_output.sc without caching, generating the following:
- .cpg.bin
- .nodes.json, .edges.json
- .dataflow.summary.json, .dataflow.*.json for each method
"""

import argparse
import sastvd.helpers.datasets as svdds
import sastvd.helpers.joern_session as svdjs
import sastvd as svd
import tqdm
from pathlib import Path


def get_dataflow_for_graph_split(worker_id, filepaths):
    sess = svdjs.JoernSession(
        f"dataflow_joern/{worker_id}",
        logfile=open(f"absdf/logs/df1g_output_{worker_id}.joernlog", "wb"),
        clean=True,
    )
    try:
        sess.import_script("get_dataflow_output")
        for fp in tqdm.tqdm(filepaths, desc="export dataflow"):
            if not Path(str(fp) + ".dataflow.json").exists():
                sess.run_script(
                    "get_dataflow_output",
                    params={"filename": fp, "cache": False},
                    import_first=False,
                )
    finally:
        sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 1G dataflow")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID to run")
    parser.add_argument("--n_splits", type=int, default=1, help="Number of splits")
    parser.add_argument(
        "--sample", action="store_true", help="Extract from only a sample"
    )
    args = parser.parse_args()

    df = svdds.bigvul(sample=args.sample)
    print("original size", len(df))

    split_len = len(df) // args.n_splits
    df = df[args.worker_id * split_len : (args.worker_id + 1) * split_len].copy()
    print(f"split", args.worker_id, len(df))

    filepaths = df["id"].apply(
        lambda _id: list(
            filter(
                Path.exists,
                (
                    svd.processed_dir() / f"bigvul/{d}/{_id}.c"
                    for d in ("before", "after")
                ),
            )
        )
    )
    filepaths = filepaths.explode()
    filepaths = filepaths[filepaths.apply(lambda fp: fp.exists())]
    print("filter to existing files", len(filepaths))
    print(filepaths)

    get_dataflow_for_graph_split(args.worker_id, filepaths)
