import sastvd.helpers.datasets as svdds
import sastvd as svd
import pandas as pd
import json

workers = 16
force = True
force_func = True

if __name__ == "__main__":
    cache_file = svd.processed_dir() / f"bigvul/1g_dataflow_hash_all.csv"

    if cache_file.exists() and not force:
        df = pd.read_csv(cache_file)
    else:
        cache_file_funcs = svd.processed_dir() / f"bigvul/1g_dataflow_funcs.csv"
        if cache_file_funcs.exists() and not force_func:
            df = pd.read_csv(cache_file_funcs)
        else:
            df = svdds.bigvul()
            df = df.rename(columns={"id": "graph_id"})[["graph_id"]]
            # df = df[df["graph_id"] == 0]
            # df = df.head(10)
            base = svd.processed_dir()
            df["summary_filepath"] = (
                df["graph_id"]
                .apply(lambda i: str(i) + ".c.dataflow.summary.json")
                .apply(lambda filename: [base/"bigvul/before"/filename, base/"bigvul/after"/filename])
                )
            df = df.explode("summary_filepath")
            df = df[df["summary_filepath"].apply(lambda p: p.exists())]
            print("summary filepath", df)

            assert len(df) > 0
            def load_summary(summary_filepath):
                with open(summary_filepath) as f:
                    return json.load(f)
            df["function"] = svd.dfmp(df, load_summary, "summary_filepath", workers=16)
            df = df.explode("function")
            print("function", df)
            df.to_csv(cache_file_funcs)

        def load_func(row):
            func = row["function"]
            with open(str(row["summary_filepath"]).replace(".dataflow.summary.json", ".dataflow." + func + ".json")) as f:
                kg = json.load(f)
            gen = {int(k): v for k, v in kg["gen"].items()}
            kill = {int(k): v for k, v in kg["kill"].items()}
            return [{
                "func": func,
                "node_id": k,
                "gen": ",".join(map(str, sorted(gen.get(k, [])))),
                "kill": ",".join(map(str, sorted(kill.get(k, []))))
            } for k in set((*gen.keys(), *kill.keys()))]
        df["genkill"] = svd.dfmp(df, load_func, ["function", "summary_filepath"], workers=16)
        df = df.explode("genkill")
        df = df.join(df["genkill"].apply(pd.Series), how='left', lsuffix="_")
        df = df[["graph_id", "func", "node_id", "gen", "kill"]].copy()
        df = df.sort_values(by=["graph_id", "func", "node_id"])
        df = df.drop_duplicates()
        print("genkill", df)

        df.to_csv(cache_file)
    print("final", df)
