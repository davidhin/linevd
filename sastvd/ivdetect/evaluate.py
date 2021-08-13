import pickle as pkl
from multiprocessing import Pool

import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.helpers as ivdh
from tqdm import tqdm


def get_dep_add_lines(filepath_before, filepath_after, added_lines):
    """Get lines that are dependent on added lines.

    Example:
    df = svdd.bigvul()
    filepath_before = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/177775.c"
    filepath_after = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/after/177775.c"
    added_lines = df[df.id==177775].added.item()

    """
    before_graph = ivdh.feature_extraction(filepath_before)[0]
    after_graph = ivdh.feature_extraction(filepath_after)[0]

    # Get nodes in graph corresponding to added lines
    added_after_lines = after_graph[after_graph.id.isin(added_lines)]

    # Get lines dependent on added lines in added graph
    dep_add_lines = added_after_lines.data.tolist() + added_after_lines.control.tolist()
    dep_add_lines = set([i for j in dep_add_lines for i in j])

    # Filter by lines in before graph
    before_lines = set(before_graph.id.tolist())
    dep_add_lines = sorted([i for i in dep_add_lines if i in before_lines])

    return dep_add_lines


def get_dep_add_lines_bigvul():
    """Cache dependent added lines for bigvul."""
    lines_dict = {}
    df = svdd.bigvul()
    df = df[df.vul == 1]
    items = df[["id", "removed", "added"]].to_dict("records")

    def helper(row):
        before_path = str(svd.processed_dir() / f"bigvul/before/{row['id']}.c")
        after_path = str(svd.processed_dir() / f"bigvul/after/{row['id']}.c")
        try:
            dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
        except Exception:
            dep_add_lines = []
        lines_dict[row["id"]] = {"removed": row["removed"], "depadd": dep_add_lines}

    pool = Pool(processes=6)
    for _ in tqdm(pool.imap_unordered(helper, items), total=len(items)):
        pass

    with open(svd.processed_dir() / "bigvul/statements/cache.pkl", "wb") as f:
        pkl.dump(lines_dict, f)
