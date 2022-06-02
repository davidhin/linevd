import pandas as pd
from pathlib import Path
import json
import re
import sastvd.helpers.joern_session as svdjs
import sastvd.helpers.joern as svdj
import tqdm
import pexpect
import sys


# %% Get all abstract dataflow info
def get_dataflow_features_df():
    dataflow_df = pd.read_csv(f"abstract_dataflow.csv")

    return dataflow_df


from sastvd.scripts.get_repos import extract_repo

checkout_dir = Path("repos/checkout")


def expand_struct_datatypes_input(df):
    md_df = pd.read_csv("bigvul_metadata_with_commit_id_slim.csv")
    md_df["repo"] = md_df["codeLink"].apply(extract_repo)
    md_df["filepath"] = md_df.apply(
        lambda row: checkout_dir
        / (
            row["repo"].replace("://", "__").replace("/", "__")
            + "__"
            + row["commit_id"]
        ),
        axis=1,
    )
    md_df = md_df[md_df["filepath"].apply(lambda p: p is not None and p.exists())]
    df = pd.merge(df, md_df, left_on="graph_id", right_on="id")
    df = df.drop_duplicates(subset=["repo", "commit_id", "datatype"]).sort_values(
        "filepath"
    )

    df = df[
        [
            "node_id",
            "graph_id",
            "project_x",
            "codeLink",
            "repo",
            "commit_id",
            "filepath",
            "datatype",
            "operator",
            "api",
            "literal",
        ]
    ].rename(columns={"project_x": "project"})

    df["datatype_subtypes"] = pd.NA
    df["datatype"] = df["datatype"].apply(
        lambda dt: dt
        if pd.isna(dt)
        else (
            re.sub(
                r"(\*|\.)",
                r"\\\1",  # escape datatype tokensregular expressions
                re.sub(
                    r"^struct ",
                    r"",  # exclude struct keyword
                    re.sub(
                        r"(\s*\*)+\s*$",
                        r"",  # exclude pointer declaration
                        re.sub(
                            r"^const ",
                            r"",  # exclude const keyword
                            re.sub(
                                r"\s+\[.*\]", r"", dt
                            ),  # exclude brackets for arrays
                        ),
                    ),
                ),
            )
        )
    )

    return df


def run_joern_adapter(df, worker_id, n_splits, test):
    cache = True
    split_len = None
    if test == "test":
        save_file = Path("bigvul_metadata_with_commit_id_slim_with_subtypes_test.csv")
        df = df.head(25)
        cache = False
    else:
        save_file = Path(
            f"bigvul_metadata_with_commit_id_slim_with_subtypes_{worker_id}.csv"
        )
        split_len = len(df) // n_splits
        print(f"{worker_id=} {n_splits=} {split_len=}")
        df = df[worker_id * split_len : (worker_id + 1) * split_len].copy()
    print(f"{test=} {len(df)=}")
    return run_joern(df, worker_id, save_file, cache)


def run_joern(df, worker_id, save_file, cache):
    sess = svdjs.JoernSession(
        f"datatype/{worker_id}",
        logfile=open(f"repos/logs/get_abs_df_{worker_id}.txt", "wb"),
    )
    sess.import_script("get_type")
    try:
        for filepath, group in tqdm.tqdm(df.groupby("filepath"), desc="load types"):
            if not sess.proc.isalive():
                raise Exception("process is no longer alive")
            try:
                dts = group["datatype"].dropna().unique()
                dt_to_subtypes = svdj.run_joern_gettype(sess, str(filepath), dts, cache)
                print(
                    "ids",
                    group["graph_id"].sort_values().unique().tolist(),
                    "filepath",
                    filepath,
                    "extracted subtypes for",
                    len(dt_to_subtypes),
                    "/",
                    len(group),
                    "datatypes",
                )
                for i, row in group.iterrows():
                    dt = row["datatype"]
                    if dt in dt_to_subtypes:
                        subtypes = dt_to_subtypes[dt]
                        print("id", i, "-", dt, "=", subtypes)
                        if len(subtypes) > 0:
                            df.at[i, "datatype_subtypes"] = subtypes
                    else:
                        print(
                            "graph_id",
                            row["graph_id"],
                            "id",
                            i,
                            "-",
                            dt,
                            "has no subtypes",
                        )
            except pexpect.exceptions.EOF:
                sess.close()
                break
    finally:
        sess.close()

    df = df.dropna(subset=["datatype_subtypes"])
    df = df.assign(
        datatype_subtypes_str=df["datatype_subtypes"].apply(
            lambda st: ", ".join(sorted(st))
        )
    )
    tdf = df.assign(
        datatype_subtypes=df["datatype_subtypes"].apply(lambda st: json.dumps(st))
    )
    tdf.to_csv(save_file)
    return df


if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    n_splits = int(sys.argv[2])
    test = sys.argv[3]
    dataflow_df = get_dataflow_features_df()
    expanded_dataflow_df_input = expand_struct_datatypes_input(dataflow_df)
    expanded_dataflow_df = run_joern_adapter(
        expanded_dataflow_df_input, worker_id, n_splits, test
    )
    merge_df = pd.merge(
        dataflow_df,
        expanded_dataflow_df[["node_id", "datatype", "datatype_subtypes_str"]],
        how="left",
        on=("node_id", "datatype"),
    )
    merge_df["datatype"] = merge_df.apply(
        lambda row: row["datatype_subtypes_str"]
        if not pd.isna(row["datatype_subtypes_str"])
        else row["datatype"],
        axis=1,
    )
