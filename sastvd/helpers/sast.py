import os
import uuid

import sastvd as svd
import sastvd.helpers.datasets as svdd
from pandarallel import pandarallel
from tqdm import tqdm

pandarallel.initialize(progress_bar=True)


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
    return ret


def rats(code: str):
    savefile = file_helper(code)
    cmd = f"rats --resultsonly --xml {savefile}"
    ret = svd.subprocess_cmd(cmd)[0].decode()
    os.remove(savefile)
    return ret


# :TODO: Fix this function.
def cppcheck(code: str):
    savefile = file_helper(code)
    cmd = (
        f"cppcheck --enable=all --inconclusive --library=posix --force --xml {savefile}"
    )
    ret = svd.subprocess_cmd(cmd)[1].decode().splitlines()
    os.remove(savefile)
    return ret


df = svdd.bigvul().sample(1000)

df["rats"] = df.before.parallel_apply(rats)
df["ffinder"] = df.before.parallel_apply(flawfinder)
df["cppcheck"] = df.before.parallel_apply(cppcheck)

allret = []
for i in tqdm(df.sample(len(df)).itertuples()):
    ret = rats(i.before)
    if len(ret) > 550:
        print(ret)
        break

allret = [i for j in allret for i in j]


[i for i in allret if "error" in i]


flawfinded = []
for i in df.itertuples():
    removed = []
    for r in i.removed:
        try:
            removed.append(int(r))
        except:
            continue
    if len(removed) == 0:
        continue
    setA = set(removed)
    setB = set(i.ffinder)
    overlap = setA & setB
    universe = setA | setB
    result1 = float(len(overlap)) / len(setA) * 100
    if result1 > 0:
        flawfinded.append(f"{i.id}: {result1}")
flawfinded
