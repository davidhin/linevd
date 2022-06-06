import sastvd.helpers.datasets as svdds
import sastvd as svd
import pandas as pd
import json
from pathlib import Path

workers = 16
force = True
sample = False

if __name__ == "__main__":
    cache_file = svd.processed_dir() / f"bigvul/1g_dataflow_hash_all_{sample}.csv"

    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file)
    else:
        df = svdds.bigvul(sample=sample)
        df = df.rename(columns={"id": "graph_id"})[["graph_id"]]
        print("start", df, sep="\n")
        base = svd.processed_dir()
        df["summary_filepath"] = (
            df["graph_id"]
            .apply(lambda i: str(i) + ".c.dataflow.json")
            .apply(
                lambda filename: [
                    base / "bigvul/before" / filename,
                    base / "bigvul/after" / filename,
                ]
            )
        )
        df = df.explode("summary_filepath")

        def check_exists(p):
            return p.exists()

        df = df[
            svd.dfmp(
                df,
                check_exists,
                "summary_filepath",
                workers=workers,
                desc="summary filepath",
            )
        ]
        print("summary filepath", df, sep="\n")

        assert len(df) > 0

        def load_summary(summary_filepath):
            with open(summary_filepath) as f:
                data = json.load(f)
            return [
                {
                    "function": funcname,
                    "genkill": [[{"node_id": node_id, "area": gk, "target": target} for node_id, target in d.items()] for gk, d in funcdata.items()],
                }
                for funcname, funcdata in data.items()
            ]

        df["dataflow"] = svd.dfmp(
            df, load_summary, "summary_filepath", workers=workers, desc="load dataflow"
        )
        df = df.explode("dataflow")
        df = df.dropna(subset=["dataflow"])
        df = pd.concat([df.drop(columns=["dataflow"]).reset_index(drop=True), pd.json_normalize(df["dataflow"], max_level=1)], axis=1)
        df = df.explode("genkill")
        df = df.explode("genkill")
        df = df.dropna(subset=["genkill"])
        df = pd.concat([df.drop(columns=["genkill"]).reset_index(drop=True), pd.json_normalize(df["genkill"])], axis=1)
        print("load dataflow 1", df, sep="\n")

        df["summary_filepath"] = df["summary_filepath"].apply(lambda fp: fp.parent.name)  # only preserve subdirectory name

        df = df.sort_values(by=["graph_id", "summary_filepath", "function", "node_id", "area"])
        print("load dataflow 2", df, sep="\n")

        df = pd.pivot(df, index=["graph_id", "summary_filepath", "function", "node_id"], columns="area", values="target").reset_index()
        df = df.rename_axis(None, axis=1)
        print("pivot", df, sep="\n")

        df["gen"] = df["gen"].apply(lambda n: json.dumps(n if isinstance(n, list) else []))
        df["kill"] = df["kill"].apply(lambda n: json.dumps(n if isinstance(n, list) else []))
        print("jsonize", df, sep="\n")

        df.to_csv(cache_file)
    print("final", df, sep="\n")
    print(df.value_counts("summary_filepath"))
    print(df["gen"].isna().value_counts().rename(index="gen.isna()"))
    print(df["kill"].isna().value_counts().rename(index="kill.isna()"))
    print(df.groupby(["graph_id", "summary_filepath"])["function"].nunique().value_counts().rename(index="unique functions per graph"))
