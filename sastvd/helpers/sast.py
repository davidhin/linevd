import os
import uuid
from xml.etree import cElementTree

import sastvd as svd


def file_helper(content: str) -> str:
    """Save content to file and return path it's saved to."""
    uid = uuid.uuid4().hex
    savefile = str(svd.cache_dir() / uid) + ".c"
    with open(savefile, "w") as f:
        f.write(content)
    return savefile


def flawfinder(code: str):
    """Run flawfinder on code string."""
    savefile = file_helper(code)
    opts = "--dataonly --quiet --singleline"
    cmd = f"flawfinder {opts} {savefile}"
    ret = svd.subprocess_cmd(cmd)[0].decode().splitlines()
    os.remove(savefile)
    records = []
    for i in ret:
        item = {"sast": "flawfinder"}
        splits = i.split(":", 2)
        item["line"] = int(splits[1])
        item["message"] = splits[2]
        records.append(item)
    return records


def rats(code: str):
    savefile = file_helper(code)
    cmd = f"rats --resultsonly --xml {savefile}"
    ret = svd.subprocess_cmd(cmd)[0].decode()
    os.remove(savefile)
    records = []
    tree = cElementTree.ElementTree(cElementTree.fromstring(ret))
    for i in tree.findall("./vulnerability"):
        item = {"sast": "rats"}
        for v in i.iter():
            if v.tag == "line":
                item["line"] = int(v.text.strip())
            if v.tag == "severity":
                item["severity"] = v.text.strip()
            if v.tag == "message":
                item["message"] = v.text.strip()
        records.append(item)
    return records


def cppcheck(code: str):
    savefile = file_helper(code)
    cmd = (
        f"cppcheck --enable=all --inconclusive --library=posix --force --xml {savefile}"
    )
    ret = svd.subprocess_cmd(cmd)[1].decode()
    os.remove(savefile)
    records = []
    tree = cElementTree.ElementTree(cElementTree.fromstring(ret))
    for i in tree.iter("error"):
        item = {"sast": "cppcheck"}
        vul_attribs = i.attrib
        loc_attribs = list(i.iter("location"))[0].attrib
        item["line"] = loc_attribs["line"]
        item["message"] = vul_attribs["msg"]
        item["severity"] = vul_attribs["severity"]
        item["id"] = vul_attribs["id"]
        records.append(item)
    return records


def run_sast(code: str, verbose: int = 0):
    rflaw = flawfinder(code)
    rrats = rats(code)
    rcpp = cppcheck(code)
    if verbose > 0:
        svd.debug(
            f"FlawFinder: {len(rflaw)} | RATS: {len(rrats)} | CppCheck: {len(rcpp)}"
        )
    return rflaw + rrats + rcpp
