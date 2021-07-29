import os
import pickle as pkl

import numpy as np
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast
from tqdm import tqdm

tqdm.pandas()

df = svdd.bigvul()
df_splits = np.array_split(df, 100)


def preprocess(row):
    """Parallelise svdj functions."""
    savedir_before = svd.get_dir(svd.processed_dir() / row["dataset"] / "before")
    savedir_after = svd.get_dir(svd.processed_dir() / row["dataset"] / "after")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.c"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.c"
    with open(fpath2, "w") as f:
        f.write(row["after"])

    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.graph.pkl"):
        joern_before = svdj.full_run_joern(fpath1)
        with open(f"{fpath1}.graph.pkl", "wb") as f:
            pkl.dump(joern_before, f)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.graph.pkl") and len(row["diff"]) > 0:
        joern_after = svdj.full_run_joern(fpath2)
        with open(f"{fpath2}.graph.pkl", "wb") as f:
            pkl.dump(joern_after, f)

    # Run SAST extraction
    fpath3 = savedir_before / f"{row['id']}.c.sast.pkl"
    if not os.path.exists(fpath3):
        sast_before = sast.run_sast(row["before"])
        with open(fpath3, "wb") as f:
            pkl.dump(sast_before, f)


for split in df_splits:
    split.progress_apply(preprocess, axis=1)
