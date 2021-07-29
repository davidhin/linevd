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
    if item["func_before"] == item["func_after"]:
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
    try:
        with open(savepath, "rb") as f:
            return pkl.load(f)
    except:
        return []


def allfunc(row, comment="before"):
    """Return a combined function (before + after commit) given the diff.

    diff = return raw diff of combined function
    added = return added line numbers relative to the combined function (start at 1)
    removed = return removed line numbers relative to the combined function (start at 1)
    before = return combined function, commented out added lines
    after = return combined function, commented out removed lines
    """
    readfile = get_codediff(row.dataset, row.id)
    if len(readfile) == 0:
        if comment == "after" or comment == "before":
            return row["func_before"]
        return []
    diff = readfile["diff"]
    if comment == "diff":
        return diff
    if comment == "added" or comment == "removed":
        return readfile[comment]
    lines = []
    for l in diff.splitlines():
        if len(l) == 0:
            continue
        if l[0] == "-":
            l = l[1:]
            if comment == "after":
                l = "// " + l
        if l[0] == "+":
            l = l[1:]
            if comment == "before":
                l = "// " + l
        lines.append(l)
    return "\n".join(lines)
