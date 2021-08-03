import re

import pandas as pd
import sastvd as svd
import sastvd.helpers.git as svdg
import sastvd.helpers.glove as svdglove
import sastvd.helpers.tokenise as svdt
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()
pandarallel.initialize(progress_bar=True, verbose=2)


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


def generate_glove(dataset="bigvul"):
    """Generate Glove embeddings for tokenised dataset."""
    if dataset == "bigvul":
        df = bigvul()

    # Only train GloVe embeddings on train samples
    samples = df.loc[df.label == "train"]

    # Preprocessing
    def get_lines(s):
        slines = s.splitlines()
        lines = []
        for sline in slines:
            tokline = svdt.tokenise(sline)
            if len(tokline) > 0:
                lines.append(tokline)
        return lines

    samples.before = samples.before.parallel_apply(get_lines)
    lines = [i for j in samples.before.to_numpy() for i in j]

    # Save corpus
    savedir = svd.get_dir(svd.processed_dir() / dataset / "glove")
    with open(savedir / "corpus.txt", "w") as f:
        f.write("\n".join(lines))

    # Train Glove Model
    CORPUS = savedir / "corpus.txt"
    svdglove.glove(CORPUS, MAX_ITER=5000)


def bigvul(minimal=True):
    """Read BigVul Data."""
    savedir = svd.get_dir(svd.cache_dir() / "minimal_datasets")
    if minimal:
        try:
            return pd.read_parquet(savedir / "minimal_bigvul.pq").dropna()
        except:
            pass
    df = pd.read_csv(svd.external_dir() / "MSR_data_cleaned.csv")
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"
    df["func_before"] = df.func_before.parallel_apply(remove_comments)
    df["func_after"] = df.func_after.parallel_apply(remove_comments)
    svdg.mp_code2diff(df)
    df = train_val_test_split_df(df, "id", "vul")
    df["added"] = df.progress_apply(svdg.allfunc, comment="added", axis=1)
    df["removed"] = df.progress_apply(svdg.allfunc, comment="removed", axis=1)
    df["diff"] = df.progress_apply(svdg.allfunc, comment="diff", axis=1)
    df["before"] = df.progress_apply(svdg.allfunc, comment="before", axis=1)
    df["after"] = df.progress_apply(svdg.allfunc, comment="after", axis=1)
    keepcols = ["dataset", "id", "label", "removed", "added", "diff", "before", "after"]
    df[keepcols].to_parquet(savedir / "minimal_bigvul.pq", index=0, compression="gzip")
    return df
