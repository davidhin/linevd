from multiprocessing import Pool, cpu_count

import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj
from tqdm import tqdm

df = svdd.bigvul()


def graph_helper(row):
    """Parallelise svdj functions."""
    svdj.full_run_joern_from_string(row["func_before"], row["dataset"], row["id"])


items = df[["id", "dataset", "func_before"]].to_dict("records")
with Pool(processes=cpu_count()) as pool:
    for _ in tqdm(pool.imap_unordered(graph_helper, items), total=len(items)):
        pass

iid = 1
path = svd.interim_dir() / f"{items[iid]['dataset']}/{items[iid]['id']}.c"
svdj.get_node_edges(path)[0]
