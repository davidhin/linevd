import pandas as pd
import re

"""
Download all repos
"""

def print_commit_ids_df():
    """
    input: bigvul_metadata.csv, MSR_data_cleaned.csv
    output: bigvul_metadata_with_commit_id.csv
    print a dataframe with commit IDs of all projects
    """
    md_df = pd.read_csv("storage/cache/bigvul/bigvul_metadata.csv")
    all_df = pd.read_csv("storage/external/MSR_data_cleaned.csv").rename(columns={"Unnamed: 0": "id"})
    all_df = all_df[["id", "project", "codeLink", "commit_id", "commit_message"]]
    df = pd.merge(md_df, all_df, on="id")
    df.to_csv("bigvul_metadata_with_commit_id.csv")

def extract_repo(cid):
    """
    input: cid (link to commit webpage as string)
    output: repo base URL (without commit ID)
    transform commit ID into repo URL
    """
    extra = ""
    if m := re.match(r"(https://github.com/[^/]+/[^/]+)", cid):
        cid = m.group(1)
        extra = "GITHUB"
    elif m := re.match(r"(.*)\?p=([^;&]+\.git)", cid):
        base_url = m.group(1)
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        git_file = m.group(2)
        cid = base_url + "/" + git_file
        extra = "PEQUALS"
    elif m := re.match(r"([^?]+\.git)", cid):
        cid = m.group(1)
        extra = "EXTRACTURL"
    elif m := re.match(r"([^?]+)/\+/", cid):
        cid = m.group(1)
        extra = "/+/"
    elif m := re.match(r"([^?]+)/commit(/|\?)", cid):
        cid = m.group(1)
        extra = "/commit/"
    elif m := re.match(r"([^?]+)/diff/", cid):
        cid = m.group(1)
        extra = "/diff/"            
    elif cid.startswith("http://git.hylafax.org/HylaFAX"):
        cid = "https://git.hylafax.org/HylaFAX.git"
        extra = "HYLAFAX"
    else:
        cid = cid
        extra = "UNPROCESSED"
    
    cid = cid.replace("%2F", "/")
    cid = cid.replace("https://cgit.freedesktop.org", "https://gitlab.freedesktop.org/")
    cid = cid.replace("https://git.haproxy.org", "http://git.haproxy.org/git")
    cid = cid.replace("https://git.gnupg.org/cgi-bin/gitweb.cgi", "https://dev.gnupg.org/source")
    cid = cid.replace("https://git.enlightenment.org/core/enlightenment.git", "https://git.enlightenment.org/enlightenment/enlightenment")
    cid = cid.replace("https://git.enlightenment.org/apps/terminology.git", "https://git.enlightenment.org/enlightenment/terminology")
    cid = cid.replace("https://git.enlightenment.org/legacy/imlib2.git", "https://git.enlightenment.org/old/legacy-imlib2")
    cid = cid.replace("https://cgit.kde.org", "https://github.com/KDE")
    # cid = cid.replace("", "")

    # import requests
    # try:
    #     status_code = str(requests.get(cid).status_code)
    # except requests.exceptions.ConnectTimeout as ex:
    #     print(ex)
    #     status_code = "timeout"
    # cid += "," + status_code

    # cid += "," + extra
    return cid

def print_repo_links():
    """
    input: bigvul_metadata_with_commit_id.csv
    output: codeLinks.txt
    print URL of each repo
    """
    df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    df["repo"] = df["codeLink"].apply(extract_repo)
    codeLinks = df["repo"].sort_values().unique().tolist()
    with open("codeLinks.txt", "w") as f:
        for cid in codeLinks:
            print(cid, file=f)

"""
1. Run download_all.sh to download repos in codeLinks.txt
"""

from pathlib import Path
import subprocess
import tqdm
import traceback
import numpy as np

base = Path("repos")
clean = base/"clean"
archive = base/"archive"

"""
2. Check out each commit from each repo.
"""

def print_uniq_repo_commit():
    """
    input: bigvul_metadata_with_commit_id.csv
    output: bigvul_metadata_with_commit_id_unique_i.csv
    split all metadata into equal-sized chunks
    """
    df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    df["repo"] = df["codeLink"].apply(extract_repo)
    df = df.sort_values(by=["repo", "commit_id"])
    df_uniq = df.drop_duplicates(subset=["repo", "commit_id"]).reset_index(drop=True)
    print("total:", len(df_uniq))
    df_uniq = df_uniq[["repo", "commit_id"]]
    df_uniq.to_csv(f"bigvul_metadata_with_commit_id_unique.csv")
    n_splits = 10
    split_size = len(df_uniq) // n_splits
    for i in range(0, n_splits):
        if i < n_splits-1:
            split = df_uniq[i*split_size:(i+1)*split_size].copy()
        else:
            split = df_uniq[i*split_size:].copy()
        print(i, len(split))
        split.to_csv(f"bigvul_metadata_with_commit_id_unique_{i}.csv")

from multiprocessing import Pool 

def archive_one_commit(row):
    repo = row.repo
    commit_id = row.commit_id
    clean_repo = row.clean_repo
    dst_repo = row.dst_repo
    prefix=f"{repo}@{commit_id}:"
    
    cmd = f"""git archive {commit_id} -o {dst_repo}"""

    out = f"{prefix}command: {cmd}"
    try:
        cmd_out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd=clean_repo, shell=True, encoding="utf-8")
        if cmd_out:
            out += "\n" + prefix + cmd_out
        return out
    except subprocess.CalledProcessError as e:
        return out+f"""{prefix}error running command "{cmd}": {traceback.format_exc()}"""


def archive_commits():
    """
    input: bigvul_metadata_with_commit_id_unique_i.csv
    output: checked-out commits in repos/checkout
    for each chunk, check out all unique commits
    """
    df = pd.read_csv(f"bigvul_metadata_with_commit_id_unique.csv")
    print("original", df)
    df.clean_repo_name = df["repo"].str.replace("://", "__").str.replace("/", "__")
    df.clean_repo = df.clean_repo_name.apply(lambda n: df.clean_repo_name/n)
    df = df[df.clean_repo.apply(lambda r: r.exists())]
    print("input exists", df)
    df.dst_repo = df.apply(lambda row: (archive/f"{row.clean_repo_name}__{row.commit_id}.tar").absolute(), axis=1)
    df = df[df.dst_repo.apply(lambda d: not d.exists())]
    print("output does not exist", df)
    df = df.head(50)  # NOTE: for test only
    with (
        Pool(10) as p,
        open(f'codeLinksCheckout_stdout.txt', 'w') as subprocess_output,
        tqdm.tqdm(p.imap_unordered(archive_one_commit, df.itertuples()), desc=f"checkout out {len(df_uniq)} commits", total=len(df_uniq)) as pbar
        ):
        for outcome in pbar:
            print(outcome, file=subprocess_output)

"""
3. Parse each repo with Joern
"""

import sastvd.helpers.joern_session as svdjs

def export_cpg(sess, fp):
    sess.run_script("get_func_graph", params={
        "filename": fp,
        "runOssDataflow": False,
        "exportJson": False,
        "exportCpg": True,
    }, import_first=False)

def parse_with_joern():
    df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    df["repo"] = df["codeLink"].apply(extract_repo)
    df["repo_filepath"] = df["repo"].replace("/", "__")
    
    sess = svdjs.JoernSession()
    sess.import_script("get_func_graph")
    try:
        df["repo_filepath"].apply(functools.partial(export_cpg, sess=sess))
    finally:
        sess.close()
