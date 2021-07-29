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

    sast_before = sast.run_sast(row["before"], verbose=1)
    with open(savedir_before / f"{row['id']}.before.sast.pkl", "wb") as f:
        pkl.dump(sast_before, f)


for split in df_splits:
    split.parallel_apply(graph_helper, axis=1)
