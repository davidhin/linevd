import os
import pickle as pkl

import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.helpers as ivdh


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


def helper(row):
    """Run get_dep_add_lines from dict.

    Example:
    df = svdd.bigvul()
    added = df[df.id==177775].added.item()
    removed = df[df.id==177775].removed.item()
    helper({"id":177775, "removed": removed, "added": added})
    """
    before_path = str(svd.processed_dir() / f"bigvul/before/{row['id']}.c")
    after_path = str(svd.processed_dir() / f"bigvul/after/{row['id']}.c")
    try:
        dep_add_lines = get_dep_add_lines(before_path, after_path, row["added"])
    except Exception:
        dep_add_lines = []
    return [row["id"], {"removed": row["removed"], "depadd": dep_add_lines}]


def get_dep_add_lines_bigvul(cache=True):
    """Cache dependent added lines for bigvul."""
    saved = svd.get_dir(svd.processed_dir() / "bigvul/eval") / "statement_labels.pkl"
    if os.path.exists(saved) and cache:
        with open(saved, "rb") as f:
            return pkl.load(f)
    df = svdd.bigvul()
    df = df[df.vul == 1]
    desc = "Getting dependent-added lines: "
    lines_dict = svd.dfmp(df, helper, ["id", "removed", "added"], ordr=False, desc=desc)
    lines_dict = dict(lines_dict)
    with open(saved, "wb") as f:
        pkl.dump(lines_dict, f)
    return lines_dict


def eval_statements(sm_logits, labels, thresh=0.5):
    """Evaluate statement-level detection according to IVDetect.

    sm_logits = [
        [0.5747372, 0.4252628],
        [0.53908646, 0.4609135],
        [0.49043426, 0.5095658],
        [0.65794635, 0.34205365],
        [0.3370166, 0.66298336],
        [0.55573744, 0.4442625],
    ]
    labels = [0, 0, 0, 0, 1, 0]
    """
    if sum(labels) == 0:
        preds = [i for i in sm_logits if i[1] > thresh]
        if len(preds) > 0:
            ret = {k: 0 for k in range(1, 11)}
        else:
            ret = {k: 1 for k in range(1, 11)}
    else:
        zipped = list(zip(sm_logits, labels))
        zipped = sorted(zipped, key=lambda x: x[0][1], reverse=True)
        ret = {}
        for i in range(1, 11):
            if 1 in [i[1] for i in zipped[:i]]:
                ret[i] = 1
            else:
                ret[i] = 0
    return ret


def eval_statements_inter(stmt_pred_list, thresh=0.5):
    """Intermediate calculation."""
    total = len(stmt_pred_list)
    ret = {k: 0 for k in range(1, 11)}
    for item in stmt_pred_list:
        eval_stmt = eval_statements(item[0], item[1], thresh)
        for i in range(1, 11):
            ret[i] += eval_stmt[i]
    ret = {k: v / total for k, v in ret.items()}
    return ret


def eval_statements_list(stmt_pred_list, thresh=0.5, vo=False):
    """Apply eval statements to whole list of preds.

    item1 = [[[0.1, 0.9], [0.6, 0.4], [0.4, 0.5]], [0, 1, 1]]
    item2 = [[[0.9, 0.1], [0.6, 0.4]], [0, 0]]
    item3 = [[[0.1, 0.9], [0.6, 0.4]], [1, 1]]
    stmt_pred_list = [item1, item2, item3]
    """
    vo_list = [i for i in stmt_pred_list if sum(i[1]) > 0]
    vulonly = eval_statements_inter(vo_list, thresh)
    if vo:
        return vulonly
    nvo_list = [i for i in stmt_pred_list if sum(i[1]) == 0]
    nonvulnonly = eval_statements_inter(nvo_list, thresh)
    ret = {}
    print(vulonly, nonvulnonly)
    for i in range(1, 11):
        ret[i] = vulonly[i] * nonvulnonly[i]
    return ret
