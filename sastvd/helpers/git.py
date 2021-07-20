import os
import pickle as pkl
import uuid
from multiprocessing import Pool, cpu_count

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
            "--no-index",
            "--no-prefix",
            f"-U{len(old.splitlines()) + len(new.splitlines())}",
            str(oldfile),
            str(newfile),
        ]
    )
    process = svd.subprocess_cmd(cmd)
    os.remove(oldfile)
    os.remove(newfile)
    return process[0].decode()


def md_lines(patch: str):
    r"""Get modified and deleted lines from Git patch.

    old = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
       asn1_push_tag(data, ASN1_GENERAL_STRING);\n\
       asn1_write_LDAPString(data, s);\n\
       asn1_pop_tag(data);\n\
       return !data->has_error;\n\
    }\n\
    \n\
    "

    new = "bool asn1_write_GeneralString(struct asn1_data *data, const char *s)\n\
    {\n\
        if (!asn1_push_tag(data, ASN1_GENERAL_STRING)) return false;\n\
        if (!asn1_write_LDAPString(data, s)) return false;\n\
        return asn1_pop_tag(data);\n\
    }\n\
    \n\
    int test() {\n\
        return 1;\n\
    }\n\
    "

    patch = gitdiff(old, new)
    """
    parsed_patch = PatchSet(patch)
    ret = {"added": [], "removed": [], "diff": ""}
    if len(parsed_patch) == 0:
        return ret
    parsed_file = parsed_patch[0]
    hunks = list(parsed_file)
    assert len(hunks) == 1
    hunk = hunks[0]
    for line in hunk:
        if line.is_added:
            ret["added"].append(line.target_line_no)
        if line.is_removed:
            ret["removed"].append(line.source_line_no)
    ret["diff"] = str(hunk).split("\n", 1)[1]
    return ret


def code2diff(old: str, new: str):
    """Get added and removed lines from old and new string."""
    patch = gitdiff(old, new)
    return md_lines(patch)


def _c2dhelper(item):
    """Given item with func_before, func_after, id, and dataset, save gitdiff."""
    savedir = svd.get_dir(svd.interim_dir() / item["dataset"])
    savepath = savedir / f"{item['id']}.git.pkl"
    if os.path.exists(savepath):
        return
    ret = code2diff(item["func_before"], item["func_after"])
    with open(savepath, "wb") as f:
        pkl.dump(ret, f)


def mp_code2diff(df):
    """Parallelise code2diff.

    Input DF must have columns: func_before, func_after, id, dataset
    """
    items = df[["func_before", "func_after", "id", "dataset"]].to_dict("records")
    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(_c2dhelper, items), total=len(items)):
            pass


def get_codediff(dataset, iid):
    """Get codediff from file."""
    savedir = svd.get_dir(svd.interim_dir() / dataset)
    savepath = savedir / f"{iid}.git.pkl"
    with open(savepath, "rb") as f:
        try:
            return pkl.load(f)
        except:
            return []
