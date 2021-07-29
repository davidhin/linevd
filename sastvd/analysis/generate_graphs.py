import os
import pickle as pkl

import numpy as np
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
import sastvd.helpers.sast as sast
from pandarallel import pandarallel

pandarallel.initialize(verbose=True, progress_bar=True)

df = svdd.bigvul()
df_splits = np.array_split(df, 100)


def graph_helper(row):
    """Parallelise svdj functions."""
    savedir_before = svd.get_dir(svd.interim_dir() / row["dataset"] / "before")
    savedir_after = svd.get_dir(svd.interim_dir() / row["dataset"] / "after")
    # svdj.full_run_joern_from_string(row["before"], row["dataset"], row["id"])

    filename = savedir_before / f"{row['id']}.before.sast.pkl"
    if os.path.exists(filename):
        return
    sast_before = sast.run_sast(row["before"])
    with open(filename, "wb") as f:
        pkl.dump(sast_before, f)


for split in df_splits:
    split.parallel_apply(graph_helper, axis=1)
