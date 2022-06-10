import os
import re

import numpy as np
import pandas as pd
import sastvd as svd
from glob import glob
from pathlib import Path
import json
import traceback
import sastvd.helpers.git as svdg
import sastvd.helpers.joern as svdj


def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)


def bigvul(cache=True, sample=False):
    """
    Read BigVul dataset from CSV
    """

    savefile = (
        svd.get_dir(svd.cache_dir() / "minimal_datasets")
        / f"minimal_bigvul{'_sample' if sample else ''}.pq"
    )
    if cache:
        try:
            df = pd.read_parquet(savefile, engine="fastparquet").dropna()

            return df
        except FileNotFoundError:
            print(f"file {savefile} not found, loading from source")
        except Exception:
            print("bigvul exception, loading from source")
            traceback.print_exc()

    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(
        svd.external_dir() / filename,
        parse_dates=["Publish Date", "Update Date"],
        dtype={
            "commit_id": str,
            "del_lines": int,
            "file_name": str,
            "lang": str,
            "lines_after": str,
            "lines_before": str,
            "Unnamed: 0": int,
            "Access Gained": str,
            "Attack Origin": str,
            "Authentication Required": str,
            "Availability": str,
            "CVE ID": str,
            "CVE Page": str,
            "CWE ID": str,
            "Complexity": str,
            "Confidentiality": str,
            "Integrity": str,
            "Known Exploits": str,
            "Score": float,
            "Summary": str,
            "Vulnerability Classification": str,
            "add_lines": int,
            "codeLink": str,
            "commit_message": str,
            "files_changed": str,
            "func_after": str,
            "func_before": str,
            "parentID": str,
            "patch": str,
            "project": str,
            "project_after": str,
            "project_before": str,
            "vul": int,
            "vul_func_with_fix": str,
        },
    )
    df = df.rename(columns={"Unnamed: 0": "id"})
    # df = df.set_index("id")
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # Save codediffs
    svd.dfmp(
        df,
        svdg._c2dhelper,
        columns=["func_before", "func_after", "id", "dataset"],
        ordr=False,
        cs=300,
    )

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)

    # POST PROCESSING
    dfv = df[df.vul == 1]
    # No added or removed but vulnerable
    dfv = dfv[~dfv.apply(lambda x: len(x.added) == 0 and len(x.removed) == 0, axis=1)]
    # Remove functions with abnormal ending (no } or ;)
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_before.strip()[-1] != "}"
            and x.func_before.strip()[-1] != ";",
            axis=1,
        )
    ]
    dfv = dfv[
        ~dfv.apply(
            lambda x: x.func_after.strip()[-1] != "}" and x.after.strip()[-1:] != ";",
            axis=1,
        )
    ]
    # Remove functions with abnormal ending (ending with ");")
    dfv = dfv[~dfv.before.apply(lambda x: x[-2:] == ");")]

    # Remove samples with mod_prop > 0.5
    dfv["mod_prop"] = dfv.apply(
        lambda x: len(x.added + x.removed) / len(x["diff"].splitlines()), axis=1
    )
    dfv = dfv.sort_values("mod_prop", ascending=0)
    dfv = dfv[dfv.mod_prop < 0.7]
    # Remove functions that are too short
    dfv = dfv[dfv.apply(lambda x: len(x.before.splitlines()) > 5, axis=1)]
    # Filter by post-processing filtering
    keep_vuln = set(dfv["id"].tolist())
    df = df[(df.vul == 0) | (df["id"].isin(keep_vuln))].copy()
    # df = df.rename(columns={"id": "example_id"})

    # df = pd.concat((
    #     df[df.vul == 0].rename(columns={"before": "code"}),
    #     df[df.vul == 1].rename(columns={"before": "code"}),
    #     df[df.vul == 1].rename(columns={"after": "code"}),
    # )).reset_index(drop=True).reset_index().rename(columns={"index": "id"})

    minimal_cols = [
        "id",
        # "example_id",
        # "code",
        "before",
        "after",
        "removed",
        "added",
        "diff",
        "vul",
        "dataset",
    ]
    df[minimal_cols].to_parquet(
        savefile,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    df[
        [
            "id",
            # "example_id",
            "commit_id",
            "vul",
            "codeLink",
            "commit_id",
            "parentID",
            "CVE ID",
            "CVE Page",
            "CWE ID",
            "Publish Date",
            "Update Date",
            "file_name",
            "files_changed",
            "lang",
            "project",
            "project_after",
            "project_before",
            "add_lines",
            "del_lines",
        ]
    ].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    return df


def check_validity(_id):
    """Check whether sample with id=_id can be loaded and has node/edges.

    Example:
    _id = 1320
    with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
        nodes = json.load(f)
    """

    valid = 0
    try:
        svdj.get_node_edges(itempath(_id))
        with open(str(itempath(_id)) + ".nodes.json", "r") as f:
            nodes = json.load(f)
            lineNums = set()
            for n in nodes:
                if "lineNumber" in n.keys():
                    lineNums.add(n["lineNumber"])
                    if len(lineNums) > 1:
                        valid = 1
                        break
            if valid == 0:
                return False
        with open(str(itempath(_id)) + ".edges.json", "r") as f:
            edges = json.load(f)
            edge_set = set([i[2] for i in edges])
            if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                return False
            return True
    except Exception as E:
        print("valid exception", traceback.format_exc(), str(itempath(_id)))
        return False


def itempath(_id):
    """Get itempath path from item id. TODO: somehow give itempath of before and after."""
    return svd.processed_dir() / f"bigvul/before/{_id}.c"


def bigvul_filter(
    df, check_file=False, check_valid=False, vulonly=False, load_code=False, sample=-1, sample_mode=False
):
    """Filter dataset based on various considerations for training"""

    # Small sample (for debugging):
    if sample > 0:
        df = df.sample(sample, random_state=0)

    # Filter only vulnerable
    if vulonly:
        df = df[df.vul == 1]

    # Filter out samples with no parsed file
    if check_file:
        finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
            if not os.path.basename(i).startswith("~")
        ]
        df = df[df.id.isin(finished)]
        print("check_file", len(df))

    # Filter out samples with no lineNumber from Joern output
    if check_valid:
        valid_cache = svd.cache_dir() / f"bigvul_valid_{sample_mode}.csv"
        if valid_cache.exists():
            valid_cache_df = pd.read_csv(valid_cache, index_col=0)
        else:
            valid = svd.dfmp(
                df, check_validity, "id", desc="Validate Samples: ", workers=6
            )
            df_id = df.id
            valid_cache_df = pd.DataFrame({"id": df_id, "valid": valid}, index=df.index)
            valid_cache_df.to_csv(valid_cache)
        df = df[df.id.isin(valid_cache_df[valid_cache_df["valid"]].id)]
        print("check_valid", len(df))

    # NOTE: drop several columns to save memory
    if not load_code:
        df = df.drop(columns=["before", "after", "removed", "added", "diff"])
    return df


def bigvul_partition(df, partition="train", undersample=True):
    """Filter to one partition of bigvul and rebalance function-wise"""

    def get_label(i):
        if i < int(len(df) * 0.1):
            return "val"
        elif i < int(len(df) * 0.2):
            return "test"
        else:
            return "train"

    df["label"] = pd.Series(
        data=list(map(get_label, range(len(df)))),
        index=np.random.RandomState(seed=0).permutation(df.index),
    )

    if partition != "all":
        df = df[df.label == partition]
    print("partitioned", len(df))

    # Balance training set
    if (partition == "train" or partition == "val") and undersample:
        vul = df[df.vul == 1]
        nonvul = df[df.vul == 0].sample(len(vul), random_state=0)
        df = pd.concat([vul, nonvul])
        print("undersampled", len(df))

    # Correct ratio for test set
    # if partition == "test" and undersample:
    #     vul = df[df.vul == 1]
    #     nonvul = df[df.vul == 0]
    #     nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0)
    #     df = pd.concat([vul, nonvul])
    #     print("undersampled", len(df))


    return df

single = {
    "api": False,
    "datatype": True,
    "literal": False,
    "operator": False,
}
all_subkeys = ["api", "datatype", "literal", "operator"]

def abs_dataflow(feat, sample=False, verbose=False):
    """Load abstract dataflow information"""
    
    df = bigvul(sample=sample)
    source_df = bigvul_partition(df, "train", undersample=False)

    abs_df_file = svd.processed_dir() / f"bigvul/abstract_dataflow_hash_api_datatype_literal_operator{'_sample' if sample else ''}.csv"
    if abs_df_file.exists():
        abs_df = pd.read_csv(abs_df_file)
        abs_df_hashes = {}
        abs_df["hash"] = abs_df["hash"].apply(json.loads)
        print(abs_df)
        # compute concatenated embedding
        for subkey in all_subkeys:
            if subkey in feat:
                if verbose:
                    print("getting hashes", subkey)
                hash_name = f"hash.{subkey}"
                abs_df[hash_name] = abs_df["hash"].apply(lambda d: d[subkey])
                if single[subkey]:
                    abs_df[hash_name] = abs_df[hash_name].apply(lambda d: d[0])
                    my_abs_df = abs_df
                else:
                    abs_df[hash_name] = abs_df[hash_name].apply(lambda d: sorted(set(d)))
                    my_abs_df = abs_df.explode(hash_name)
                if verbose:
                    vc = my_abs_df[hash_name].value_counts()
                    print(vc)
                    print(len(vc.loc[vc > 1].index), "more than 1")
                    print(len(vc.loc[vc > 5].index), "more than 5")
                    print(len(vc.loc[vc > 100].index), "more than 100")
                    print(len(vc.loc[vc > 1000].index), "more than 1000")
                my_abs_df = my_abs_df[["graph_id", "node_id", "hash", hash_name]]
                
                hashes = pd.merge(source_df, my_abs_df, left_on="id", right_on="graph_id")[hash_name].dropna()
                # most frequent
                if verbose:
                    print("min", hashes.value_counts().head(1000).min(), hashes.value_counts().head(1000).idxmin())
                hashes = hashes.value_counts().head(1000).index.sort_values().unique().tolist()
                hashes.insert(0, None)

                abs_df_hashes[subkey] = {h: i for i, h in enumerate(hashes)}

                print("trained hashes", subkey, len(abs_df_hashes[subkey]))

        if "all" in feat:
            source_df_hashes = pd.merge(source_df, abs_df, left_on="id", right_on="graph_id")
            def get_all_hash(row):
                h = {}
                for subkey in all_subkeys:
                    if subkey in feat:
                        hash_name = f"hash.{subkey}"
                        hashes = abs_df_hashes[subkey]
                        hash_values = row[hash_name]
                        if "includeunknown" in feat:
                            if single[subkey]:
                                hash_idx = [hash_values]
                            else:
                                hash_idx = hash_values
                        else:
                            if single[subkey]:
                                hash_idx = [hash_values if hash_values in hashes else "UNKNOWN"]
                            else:
                                hash_idx = [hh if hh in hashes else "UNKNOWN" for hh in hash_values]
                        # print(hash_idx)
                        h[subkey] = sorted(set(hash_idx))
                return h
            abs_df["hash.all"] = source_df_hashes.apply(get_all_hash, axis=1).apply(json.dumps)
            if verbose:
                vc = abs_df.value_counts("hash.all")
                print(vc)
                print(len(vc.loc[vc > 1].index), "more than 1")
                print(len(vc.loc[vc > 5].index), "more than 5")
                print(len(vc.loc[vc > 100].index), "more than 100")
                print(len(vc.loc[vc > 1000].index), "more than 1000")
                print("min", vc.head(1000).min(), vc.head(1000).idxmin())
            all_hashes = abs_df["hash.all"].value_counts().head(1000).index.sort_values().unique().tolist()
            all_hashes.insert(0, None)
            abs_df_hashes["all"] = {h: i for i, h in enumerate(all_hashes)}

        return abs_df, abs_df_hashes
    else:
        print("YOU SHOULD RUN `python sastvd/scripts/abstract_dataflow_full.py --stage 2`")

def test_abs():
    abs_df, abs_df_hashes = abs_dataflow(feat="_ABS_DATAFLOW_api_datatype_literal_operator", sample=False, verbose=True)
    assert all(not all(abs_df[f"hash.{subkey}"].isna()) for subkey in all_subkeys)
    assert len([c for c in abs_df.columns if "hash." in c]) == len(all_subkeys)
    assert len(abs_df_hashes) == len(all_subkeys)

def test_abs_all():
    for featname in ("datatype", "literal_operator", "api_literal_operator", "api_datatype_literal_operator_all"):
        print(featname)
        abs_df, abs_df_hashes = abs_dataflow(feat=f"_ABS_DATAFLOW_{featname}_all", sample=False)
        vc = abs_df.value_counts("hash.all")
        print(vc)
        print(len(vc.loc[vc > 1].index), "more than 1")
        print(len(vc.loc[vc > 5].index), "more than 5")
        print(len(vc.loc[vc > 100].index), "more than 100")
        print(len(vc.loc[vc > 1000].index), "more than 1000")
        print("min", vc.head(1000).min(), vc.head(1000).idxmin())


def test_abs_all_unk():
    for featname in ("datatype", "literal_operator", "api_literal_operator", "api_datatype_literal_operator_all"):
        print(featname)
        abs_df, abs_df_hashes = abs_dataflow(feat=f"_ABS_DATAFLOW_{featname}_all_includeunknown", sample=False)
        vc = abs_df.value_counts("hash.all")
        print(vc)
        print(len(vc.loc[vc > 1].index), "more than 1")
        print(len(vc.loc[vc > 5].index), "more than 5")
        print(len(vc.loc[vc > 100].index), "more than 100")
        print(len(vc.loc[vc > 1000].index), "more than 1000")
        print("min", vc.head(1000).min(), vc.head(1000).idxmin())


def dataflow_1g(sample=False):
    """Load 1st generation dataflow information"""

    cache_file = svd.processed_dir() / f"bigvul/1g_dataflow_hash_all_{sample}.csv"
    if cache_file.exists():
        df = pd.read_csv(
            cache_file,
            converters={
                "graph_id": int,
                "node_id": int,
                "func": str,
                "gen": str,
                "kill": str,
            },
        )
        df["gen"] = df["gen"].apply(json.loads)
        df["kill"] = df["kill"].apply(json.loads)
        return df
    else:
        print("YOU SHOULD RUN dataflow_1g.py")

def test_1g():
    print(dataflow_1g(sample=True))
