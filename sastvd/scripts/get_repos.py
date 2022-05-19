import pandas as pd
import re

def get_repos():
    md_df = pd.read_csv("storage/cache/bigvul/bigvul_metadata.csv")
    all_df = pd.read_csv("storage/external/MSR_data_cleaned.csv").rename(columns={"Unnamed: 0": "id"})
    all_df = all_df[["id", "project", "codeLink", "commit_id", "commit_message"]]
    df = pd.merge(md_df, all_df, on="id")
    df.to_csv("bigvul_metadata_with_commit_id.csv")

def extract_repo(cid):
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

def print_repos():
    df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    df["repo"] = df["codeLink"].apply(extract_repo)
    codeLinks = df["repo"].sort_values().unique().tolist()
    with open("codeLinks.txt", "w") as f:
        for cid in codeLinks:
            print(cid, file=f)

# run download_all.sh on codeLinks.txt

from pathlib import Path
import subprocess
import tqdm
import traceback
import numpy as np

base = Path("repos")
clean = base/"clean"
checkout = base/"checkout"

def get_repos_commits_split():
    df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    df["repo"] = df["codeLink"].apply(extract_repo)
    df = df.sort_values(by=["repo", "commit_id"])
    df_uniq = df.groupby(['repo','commit_id']).size().reset_index()
    n_splits = 10
    split_size = len(df) // n_splits
    for i in range(0, n_splits):
        if i < n_splits-1:
            split = df[i*split_size:(i+1)*split_size].copy()
        else:
            split = df[i*split_size:].copy()
        split.to_csv(f"bigvul_metadata_with_commit_id_unique_{i}.csv")
        

def get_repos_commits(i):
    df_uniq = pd.read_csv(f"bigvul_metadata_with_commit_id_unique_{i}.csv")
    print(df_uniq)
    # df_uniq = df_uniq.head(5)  # NOTE: for test only
    with (
        open(f'codeLinksCheckout_stdout_{i}.txt', 'w') as subprocess_output,
        tqdm.tqdm(df_uniq.iterrows(), desc=f"checkout out {len(df_uniq)} commits", total=len(df_uniq)) as pbar
        ):
        for i, row in pbar:
            repo = row["repo"]
            commit_id = row["commit_id"]
            
            clean_repo_name = repo.replace("/", "__")
            clean_repo = clean/clean_repo_name
            if not clean_repo.exists():
                print(f"{clean_repo} does not exist. Skipping...")
                continue
            dst_repo = (checkout/f"{clean_repo_name}____{commit_id}").absolute()
            if dst_repo.exists():
                print(f"{dst_repo} exists. Skipping...")
                continue
            cmd = f"""git worktree add -f -f {dst_repo} {commit_id}"""

            print(f"command: {cmd}", file=subprocess_output, flush=True)
            try:
                subprocess.check_call(cmd, stdout=subprocess_output, stderr=subprocess.STDOUT, cwd=clean_repo, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"""error running command "{cmd}": {traceback.format_exc()}""")
