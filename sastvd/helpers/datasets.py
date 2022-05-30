import os
import re

import numpy as np
import pandas as pd
import sastvd as svd
from glob import glob
from pathlib import Path
import json
import traceback
import sastvd.helpers.doc2vec as svdd2v
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split


def train_val_test_split_df(df, idcol, labelcol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[labelcol]
    train_rat = 0.8
    val_rat = 0.1
    test_rat = 0.1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_rat, random_state=1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=test_rat / (test_rat + val_rat), random_state=1
    )
    X_train = set(X_train)
    X_val = set(X_val)
    X_test = set(X_test)

    def path_to_label(path):
        if path in X_train:
            return "train"
        if path in X_val:
            return "val"
        if path in X_test:
            return "test"

    df["label"] = df[idcol].apply(path_to_label)
    return df


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


def generate_glove(dataset="bigvul", sample=False, cache=True):
    """Generate Glove embeddings for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    if os.path.exists(savedir / "vectors.txt") and cache:
        svd.debug("Already trained GloVe.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)
    MAX_ITER = 2 if sample else 500

    # Only train GloVe embeddings on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Save corpus
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"glove_{sample}")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # Train Glove Model
    CORPUS = savedir / "corpus.txt"
    svdglove.glove(CORPUS, MAX_ITER=MAX_ITER)


def generate_d2v(dataset="bigvul", sample=False, cache=True, **kwargs):
    """Train Doc2Vec model for tokenised dataset."""
    savedir = svd.get_dir(svd.processed_dir() / dataset / f"d2v_{sample}")
    if os.path.exists(savedir / "d2v.model") and cache:
        svd.debug("Already trained Doc2Vec.")
        return
    if dataset == "bigvul":
        df = bigvul(sample=sample)

    # Only train Doc2Vec on train samples
    samples = df[df.label == "train"].copy()

    # Preprocessing
    samples.before = svd.dfmp(
        samples, svdt.tokenise_lines, "before", cs=200, desc="Get lines: "
    )
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Train Doc2Vec model
    model = svdd2v.train_d2v(lines, **kwargs)

    # Test Most Similar
    most_sim = model.dv.most_similar([model.infer_vector("memcpy".split())])
    for i in most_sim:
        print(lines[i[0]])
    model.save(str(savedir / "d2v.model"))


def bigvul(cache=True, sample=False, return_raw=False, splits="default", after_parse=True):
    """Read BigVul Data.

    Args:
        sample (bool): Only used for testing!
        splits (str): default, crossproject-(linux|Chrome|Android|qemu)

    EDGE CASE FIXING:
    id = 177860 should not have comments in the before/after
    """

    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    if cache:
        try:
            df = pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()

            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            md.groupby("project").count().sort_values("id")

            #default_splits = svd.external_dir() / "bigvul_rand_splits.csv"
            #if os.path.exists(default_splits):
            #    splits = pd.read_csv(default_splits)
            #    splits = splits.set_index("id").to_dict()["label"]
            #    df["label"] = df.id.map(splits)
            
            """
            def get_label(i):
                if i < int(len(df) * 0.1):
                    return "val"
                elif i < int(len(df) * 0.2):
                    return "test"
                else:
                    return "train"
            df["label"] = pd.Series(data=list(map(get_label, range(len(df)))), index=np.random.RandomState(seed=0).permutation(df.index))
            print("splits", df["label"].value_counts())
            """

            if "crossproject" in splits:
                raise NotImplementedError(splits)
                project = splits.split("_")[-1]
                md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
                nonproject = md[md.project != project].id.tolist()
                trid, vaid = train_test_split(nonproject, test_size=0.1, random_state=1)
                teid = md[md.project == project].id.tolist()
                teid = {k: "test" for k in teid}
                trid = {k: "train" for k in trid}
                vaid = {k: "val" for k in vaid}
                cross_project_splits = {**trid, **vaid, **teid}
                df["label"] = df.id.map(cross_project_splits)

            return df
        except Exception as E:
            print("bigvul exception", E)

            pass
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(svd.external_dir() / filename, parse_dates=['Publish Date', 'Update Date'], dtype={
        'commit_id': str,
        'del_lines': int,
        'file_name': str,
        'lang': str,
        'lines_after': str,
        'lines_before': str,

        'Unnamed: 0': int,
        'Access Gained': str,
        'Attack Origin': str,
        'Authentication Required': str,
        'Availability': str,
        'CVE ID': str,
        'CVE Page': str,
        'CWE ID': str,
        'Complexity': str,
        'Confidentiality': str,
        'Integrity': str,
        'Known Exploits': str,
        # 'Publish Date': pd.datetime64,
        'Score': float,
        'Summary': str,
        # 'Update Date': pd.datetime64,
        'Vulnerability Classification': str,
        'add_lines': int,
        'codeLink': str,
        'commit_message': str,
        'files_changed': str,
        'func_after': str,
        'func_before': str,
        'parentID': str,
        'patch': str,
        'project': str,
        'project_after': str,
        'project_before': str,
        'vul': int,
        'vul_func_with_fix': str,
    })
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # Return raw (for testing)
    if return_raw:
        return df

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)

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
    keep_vuln = set(dfv.id.tolist())
    df = df[(df.vul == 0) | (df.id.isin(keep_vuln))].copy()

    # Make splits
    df = train_val_test_split_df(df, "id", "vul")

    keepcols = [
        "dataset",
        "id",
        "label",
        "removed",
        "added",
        "diff",
        "before",
        "after",
        "vul",
    ]
    df_savedir = savedir / f"minimal_bigvul_{sample}.pq"
    df[keepcols].to_parquet(
        df_savedir,
        object_encoding="json",
        index=0,
        compression="gzip",
        engine="fastparquet",
    )
    metadata_cols = df.columns[:17].tolist() + ["project"]
    df[metadata_cols].to_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv", index=0)
    return df



def check_validity(_id):
    """Check whether sample with id=_id has node/edges.

    Example:
    _id = 1320
    with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
        nodes = json.load(f)
    """
    import sastvd.helpers.joern as svdj
    
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


def bigvul_filter(df, check_file=False, check_valid=False, vulonly=False, load_code=False, sample=-1):
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
        valid_cache = svd.cache_dir() / f"bigvul_valid.csv"
        if valid_cache.exists():
            valid_cache_df = pd.read_csv(valid_cache, index_col=0)
        else:
            valid = svd.dfmp(
                df, check_validity, "id", desc="Validate Samples: ",
                workers=6
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

def bigvul_partition(df, partition="train"):
    """Filter to one partition of bigvul and rebalance function-wise"""

    def get_label(i):
        if i < int(len(df) * 0.1):
            return "val"
        elif i < int(len(df) * 0.2):
            return "test"
        else:
            return "train"
    df["label"] = pd.Series(data=list(map(get_label, range(len(df)))), index=np.random.RandomState(seed=0).permutation(df.index))
    # print("splits", df["label"].value_counts())

    # breakpoint()
    df = df[df.label == partition]
    # print("len(df)=", len(df))
    # print("df head=", df.head())

    # Balance training set
    if partition == "train" or partition == "val":
        vul = df[df.vul == 1]
        nonvul = df[df.vul == 0].sample(len(vul), random_state=0)
        df = pd.concat([vul, nonvul])

    # Correct ratio for test set
    if partition == "test":
        vul = df[df.vul == 1]
        nonvul = df[df.vul == 0]
        nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0)
        df = pd.concat([vul, nonvul])
    
    return df

def abs_dataflow():
    """Load abstract dataflow information"""
    
    abs_df_file = svd.processed_dir() / f"bigvul/abstract_dataflow_hash_all.csv"
    if abs_df_file.exists():
        abs_df = pd.read_csv(abs_df_file)
        abs_df["hash"] = abs_df["hash"].fillna(-1)
        abs_df_hashes = sorted(abs_df["hash"].unique().tolist())
        # abs_df_hashes.insert(0, -1)
        abs_df_hashes.insert(0, None)
    else:
        print("YOU SHOULD RUN abstract_dataflow.py")
    return abs_df, abs_df_hashes

def dataflow_1g():
    """Load 1st generation dataflow information"""
    
    cache_file = svd.processed_dir() / f"bigvul/1g_dataflow_hash_all.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        print("YOU SHOULD RUN dataflow_1g.py")
    return df

def bigvul_cve():
    """Return id to cve map."""
    md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
    ret = md[["id", "CVE ID"]]
    return ret.set_index("id").to_dict()["CVE ID"]
