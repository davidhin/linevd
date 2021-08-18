import json
import os
import re
from glob import glob
from pathlib import Path

import pandas as pd
import sastvd as svd
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from sklearn.model_selection import train_test_split


def train_val_test_split_df(df, idcol, stratifycol):
    """Add train/val/test column into dataframe."""
    X = df[idcol]
    y = df[stratifycol]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
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


def bigvul(minimal=True, sample=False):
    """Read BigVul Data.

    Args:
        sample (bool): Only used for testing!

    EDGE CASE FIXING:
    id = 177775 should have removed line = 16 and added line = 17
    id = 177860 should not have comments in the before/after
    """
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    if minimal:
        try:
            return pd.read_parquet(
                savedir / f"minimal_bigvul_{sample}.pq", engine="fastparquet"
            ).dropna()
        except:
            pass
    filename = "MSR_data_cleaned_SAMPLE.csv" if sample else "MSR_data_cleaned.csv"
    df = pd.read_csv(svd.external_dir() / filename)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = svd.dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = svd.dfmp(df, remove_comments, "func_after", cs=500)

    # Save codediffs
    cols = ["func_before", "func_after", "id", "dataset"]
    svd.dfmp(df, svdg._c2dhelper, columns=cols, ordr=False, cs=300)

    # Make splits
    df = train_val_test_split_df(df, "id", "vul")

    # Assign info and save
    df["info"] = svd.dfmp(df, svdg.allfunc, cs=500)
    df = pd.concat([df, pd.json_normalize(df["info"])], axis=1)
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
    return df


class BigVulDataset:
    """Represent BigVul as graph dataset."""

    def __init__(self, partition="train", sample=-1):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
        ]
        self.df = bigvul()
        self.partition = partition
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]

        # Balance training set
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter out samples with no lineNumber from Joern output
        self.df["valid"] = svd.dfmp(
            self.df, BigVulDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        # Load Glove vectors.
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.emb_dict, _ = svdglove.glove_dict(glove_path)

    def itempath(_id):
        """Get itempath path from item id."""
        return svd.processed_dir() / f"bigvul/before/{_id}.c"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        with open(str(BigVulDataset.itempath(_id)) + ".nodes.json", "r") as f:
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
        with open(str(BigVulDataset.itempath(_id)) + ".edges.json", "r") as f:
            edges = json.load(f)
            edge_set = set([i[2] for i in edges])
            if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                return False
            return True

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])

    def __getitem__(self, idx):
        """Must override."""
        return self.df[self.df.idx == idx].to_dict("records")[0]

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"
