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


def slug(cid):
    return cid.replace("://", "__").replace("//", "__").replace("/", "__")

def correct_repo_name(cid):
    bigmap = {
        "http://git.infradead.org/users/dwmw2/openconnect.git":
        "git://git.infradead.org/users/dwmw2/openconnect.git",
        "http://git.infradead.org/users/steved/libtirpc.git":
        "git://git.infradead.org/users/dwmw2/openconnect.git",
        "http://git.linux-nfs.org/steved/libtirpc.git":
        "git://git.linux-nfs.org/projects/steved/libtirpc.git",
        "http://git.savannah.nongnu.org/cgit/exosip.git":
        "https://git.savannah.gnu.org/git/exosip.git",
        "http://git.wpitchoune.net/gitweb/psensor.git":
        "https://gitlab.com/jeanfi/psensor.git",
        "https://git.savannah.gnu.org/cgit/quagga.git":
        "https://github.com/Quagga/quagga",
        "https://git.savannah.gnu.org/gitweb/quagga.git":
        "https://github.com/Quagga/quagga",
        "https://git.enlightenment.org/apps/terminology.git":
        "http://git.enlightenment.org/enlightenment/terminology.git",
        "https://git.enlightenment.org/core/enlightenment.git":
        "http://git.enlightenment.org/enlightenment/enlightenment.git",
        "https://git.enlightenment.org/legacy/imlib2.git":
        "http://git.enlightenment.org/old/legacy-imlib2.git",
        "https://git.exim.org/exim.git":
        "git://git.exim.org/exim.git",
        "https://git.gnupg.org/cgi-bin/gitweb.cgi/gnupg.git":
        "git://git.gnupg.org/gnupg.git",
        "https://git.gnupg.org/cgi-bin/gitweb.cgi/gpgme.git":
        "git://git.gnupg.org/gpgme.git",
        "https://git.gnupg.org/cgi-bin/gitweb.cgi/libgcrypt.git":
        "git://git.gnupg.org/libgcrypt.git",
        "https://git.gnupg.org/cgi-bin/gitweb.cgi/libksba.git":
        "git://git.gnupg.org/libksba.git",
        "https://git.hylafax.org/HylaFAX.git":
        "https://github.com/ifax/HylaFAX.git",
        "https://git.lxde.org/gitweb/lxde/lxterminal.git":
        "https://github.com/lxde/lxterminal.git",
        "https://git.lxde.org/gitweb/lxde/menu-cache.git":
        "https://github.com/lxde/menu-cache.git",
        "https://git.lxde.org/gitweb/lxde/pcmanfm.git":
        "https://github.com/lxde/pcmanfm.git",
        "https://github.com/php/php-src.git.git":
        "https://github.com/php/php-src.git",
        "https://github.com/php/php-src":
        "https://github.com/php/php-src.git",
        "https://git.php.net/php-src.git":
        "https://github.com/php/php-src.git",
        "https://git.postgresql.org/gitweb/postgresql.git":
        "https://github.com/postgres/postgres",
        "https://git.qemu.org/qemu.git":
        "https://git.qemu.org/git/qemu.git",
        "https://git.qemu.org/gitweb.cgi/qemu.git":
        "https://git.qemu.org/git/qemu.git",
        "https://github.com/tomhughes/libdwarf":
        "https://github.com/davea42/libdwarf-code",
        "https://git.shibboleth.net/view/cpp-opensaml.git":
        "https://git.shibboleth.net/git/cpp-opensaml.git",
        "https://git.shibboleth.net/view/cpp-sp.git":
        "https://git.shibboleth.net/git/cpp-sp.git",
        "https://git.shibboleth.net/view/cpp-xmltooling.git":
        "https://git.shibboleth.net/git/cpp-xmltooling.git",
        "https://git.openssl.org/gitweb/openssl.git":
        "git://git.openssl.org/openssl.git",
        "https://git.openssl.org/openssl.git":
        "git://git.openssl.org/openssl.git",
        # multiple occurrences of rsync
        "https://git.samba.org/rsync.git/rsync.git":
        "https://git.samba.org/rsync.git",
        "https://git.quassel-irc.org/quassel.git":
        "https://github.com/quassel/quassel",
        "https://htcondor-git.cs.wisc.edu/condor.git":
        "https://github.com/htcondor/htcondor",
        # misc freedesktop crap
        "https://gitlab.freedesktop.org/accountsservice":
        "https://gitlab.freedesktop.org/accountsservice/accountsservice",
        "https://gitlab.freedesktop.org/drm/drm-misc":
        "https://gitlab.freedesktop.org/drm/misc",
        "https://gitlab.freedesktop.org/exempi":
        "https://gitlab.freedesktop.org/libopenraw/exempi",
        "https://gitlab.freedesktop.org/fontconfig":
        "https://gitlab.freedesktop.org/fontconfig/fontconfig",
        "https://gitlab.freedesktop.org/harfbuzz":
        "https://github.com/freedesktop/harfbuzz",
        "https://gitlab.freedesktop.org/harfbuzz.old":
        "git://anongit.freedesktop.org/harfbuzz.old",
        "https://gitlab.freedesktop.org/libbsd":
        "https://gitlab.freedesktop.org/libbsd/libbsd",
        "https://gitlab.freedesktop.org/pixman":
        "https://gitlab.freedesktop.org/pixman/pixman",
        "https://gitlab.freedesktop.org/polkit":
        "https://gitlab.freedesktop.org/polkit/polkit",
        "https://gitlab.freedesktop.org/systemd/systemd":
        "https://github.com/freedesktop/systemd",
        "https://gitlab.freedesktop.org/udisks":
        "https://github.com/storaged-project/udisks",
        "https://gitlab.freedesktop.org/virglrenderer":
        "https://gitlab.freedesktop.org/virgl/virglrenderer",
    }
    if cid in bigmap:
        cid = bigmap[cid]
    # replace cgit with gitlab
    cid = cid.replace("cgit.", "gitlab.")
    # replace cgit with github
    cid = cid.replace("cgit.kde.org", "github.com/KDE").replace("gitlab.kde.org", "github.com/KDE")
    # replace git.savannah.gnu.org/*/ with git.savannah.gnu.org/git
    cid = (
        cid
        .replace("git.savannah.gnu.org/cgit", "git.savannah.gnu.org/git")
        .replace("git.savannah.gnu.org/gitview", "git.savannah.gnu.org/git")
        .replace("git.savannah.gnu.org/gitweb", "git.savannah.gnu.org/git")
    )

    # misc mappings
    cid = cid.replace("https://dev.gnupg.org/source", "git://git.gnupg.org")
    cid = cid.replace("git.haproxy.org", "git.haproxy.org/git")
    cid = cid.replace("git.gnupg.org/cgi-bin/gitweb.cgi", "git.gnupg.org")

    return cid

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
    cid = correct_repo_name(cid)

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
    df["repo"] = df["codeLink"].map(extract_repo)
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
archive = base/"archive3"
# checkout = base/"checkout2"

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
    # n_splits = 10
    # split_size = len(df_uniq) // n_splits
    # for i in range(0, n_splits):
    #     if i < n_splits-1:
    #         split = df_uniq[i*split_size:(i+1)*split_size].copy()
    #     else:
    #         split = df_uniq[i*split_size:].copy()
    #     print(i, len(split))
    #     split.to_csv(f"bigvul_metadata_with_commit_id_unique_{i}.csv")

from multiprocessing import Pool 

def archive_one_commit(row):
    repo = row["repo"]
    commit_id = row["commit_id"]
    clean_repo = row["clean_repo"]
    dst_repo = row["dst_repo"]
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
    df["repo"] = df["repo"].map(correct_repo_name)
    df["clean_repo_name"] = df["repo"].map(slug)
    
    df["clean_repo"] = df["clean_repo_name"].apply(lambda n: clean/n)
    df = df[df["clean_repo"].apply(lambda r: r.exists())]
    print("input exists", df)
    print("unique items", len(df["clean_repo_name"].unique()))
    # print("\n".join(df["clean_repo_name"].sort_values().unique().tolist()))

    df["dst_repo"] = df.apply(lambda row: (archive/f"{row['clean_repo_name']}__{row['commit_id']}.tar").absolute(), axis=1)
    df = df[df["dst_repo"].apply(lambda d: not d.exists())]
    df = df[df.apply(lambda row: not (base/"archive"/f"{row['clean_repo_name']}__{row['commit_id']}.tar").absolute().exists(), axis=1)]
    print("output does not exist", df)

    # return
    # df = df.head(50)  # NOTE: for test only
    with (
        Pool(6) as p,
        open(f'codeLinksCheckout_stdout3.txt', 'w') as subprocess_output,
        tqdm.tqdm(p.imap_unordered(archive_one_commit, df.to_dict("records")), desc=f"archive {len(df)} commits", total=len(df)) as pbar
        ):
        for outcome in pbar:
            print(outcome, file=subprocess_output)

import shutil
import tarfile

def get_checkout_dir(tf):
    return checkout/tf.stem
def extract(tf):
    cd = get_checkout_dir(tf)
    try:
        cd.mkdir(exist_ok=True)
        with tarfile.open(tf) as my_tar:
            my_tar.extractall(cd)
    except Exception:
        print(tf, "->", cd, "error:", traceback.format_exc())
    return cd

def extract_archived_commits(split_idx=-1, n_splits=1):
    tar_files = [Path(l.strip()) for l in open(archive/"index_fixed.txt").readlines() if l]
    print(len(tar_files), "tar files")
    to_extract = []
    for tf in tqdm.tqdm(tar_files, desc="check existing"):
        if True:
            to_extract.append(tf)
    if split_idx == -1:
        to_extract = to_extract[:5]  # NOTE: test only
    else:
        splits = np.array_split(to_extract, n_splits)
        to_extract = splits[split_idx]
    print(len(to_extract), "to extract")
    try:
        for cd in tqdm.tqdm(map(extract, to_extract), total=len(to_extract), desc="untarring files"):
            pass
    except:
        raise

"""
3. Parse each repo with Joern
"""

import functools

import sastvd.helpers.joern_session as svdjs

# repos_path = Path("repos/checkout3")

def export_cpg(sess, fp):
    try:
        return sess.run_script("get_func_graph", params={
            "filename": fp,
            "runOssDataflow": False,
            "exportJson": False,
            "exportCpg": True,
        }, import_first=False, timeout=60*10)
    except Exception:
        return traceback.format_exc()

def parse_with_joern(job_array_id=-1, n_splits=1, repos_path="repos/checkout3", stuff_idx=3):
    repos_path = Path(repos_path)
    df = pd.read_csv(f"bigvul_metadata_with_commit_id_unique.csv")
    df["repo"] = df["repo"].map(correct_repo_name)
    df["clean_repo_name"] = df["repo"].map(slug)
    # df = pd.read_csv("bigvul_metadata_with_commit_id.csv")
    # df["repo"] = df["codeLink"].apply(extract_repo)
    df["repo_filepath"] = df["clean_repo_name"].apply(lambda r: str(repos_path/(r)))
    df["checkout_filepath"] = df[['repo_filepath', 'commit_id']].T.agg('__'.join)
    print("original", df)
    df = df[df["checkout_filepath"].apply(lambda fp: Path(fp).exists())]
    print("input exists", df)
    df["checkout_filepath_dst"] = df["checkout_filepath"].apply(lambda fp: fp + ".cpg.bin")
    df = df[df["checkout_filepath_dst"].apply(lambda fp: not Path(fp + ".cpg.bin").exists())]
    print("output does not exist", df)

    # split
    split_size = len(df) // n_splits
    logfile = open(f"output_parse_{job_array_id}.txt", "wb")
    if job_array_id == -1:
        df = df.head(5)  # NOTE: debug
    elif job_array_id == n_splits:
        df = df[job_array_id*split_size:].copy()
    else:
        df = df[job_array_id*split_size:(job_array_id+1)*split_size].copy()
    
    sess = svdjs.JoernSession(f"cpg{stuff_idx}_{job_array_id}", logfile=logfile, clean=True)
    sess.import_script("get_func_graph")
    try:
        export_output = []
        for fp in tqdm.tqdm(df["checkout_filepath"], desc="extract CPG"):
            export_output.append(export_cpg(sess, fp))
        df["export_output"] = export_output
        print(df["export_output"].iloc[0])
    finally:
        sess.delete()
        sess.close()
    df.to_csv(f"bigvul_metadata_with_commit_id_parse_{job_array_id}.csv")

# if __name__ == "__main__":
#     import sys
#     parse_with_joern()
