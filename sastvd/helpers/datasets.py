import os
import uuid
from multiprocessing import Pool, cpu_count

import pandas as pd
import sastvd as svd
from tqdm import tqdm
from unidiff import PatchSet


def gitdiff(old: str, new: str):
    """Git diff between two strings."""
    cachedir = svd.cache_dir()
    oldfile = cachedir / uuid.uuid4().hex
    newfile = cachedir / uuid.uuid4().hex
    with open(oldfile, "w") as f:
        f.write(old)
    with open(newfile, "w") as f:
        f.write(new)
    cmd = " ".join(
        [
            "git",
            "diff",
            "--unified=0",
            str(oldfile),
            str(newfile),
        ]
    )
    process = svd.subprocess_cmd(cmd)
    os.remove(oldfile)
    os.remove(newfile)
    return process[0].decode()


def md_lines(patch: str):
    """Get modified and deleted lines from Git patch."""
    parsed_patch = PatchSet(patch)
    ret = {"added": [], "removed": []}
    for parsed_file in parsed_patch:
        for hunk in parsed_file:
            for line in hunk:
                if line.is_added:
                    ret["added"].append(line.target_line_no)
                if line.is_removed:
                    ret["removed"].append(line.source_line_no)
    return ret


def code2diff(old: str, new: str):
    """Get added and removed lines from old and new string."""
    patch = gitdiff(old, new)
    return md_lines(patch)


def mp_code2diff(df):
    """Parallelise code2diff.

    Input DF must have columns: func_before, func_after, id, dataset
    """
    items = df[["func_before", "func_after"]].to_dict("records")

    def c2dhelper(item):
        code2diff(item["func_before"], item["func_after"])

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(c2dhelper, items), total=len(items)):
            pass


def bigvul():
    """Read BigVul Data."""
    df = pd.read_csv(svd.external_dir() / "bigvul2020.csv.gzip", compression="gzip")
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"
    return df
