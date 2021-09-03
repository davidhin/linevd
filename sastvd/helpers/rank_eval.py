import numpy as np
from sklearn.metrics import roc_auc_score


def precision_at_k(r, k):
    """Calculate precision @ k.

    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r, limit):
    """Calculate average precision (area under PR curve).

    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(limit) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs, k):
    """Calculate mean average precision.

    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r, k) for r in rs])


def dcg_at_k(r, k, method=0):
    """Calculate discounted cumulative gain (dcg).

    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=0):
    """Calculate normalized discounted cumulative gain (ndcg).

    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    r_ = r[0:k]
    dcg_max = dcg_at_k(sorted(r_, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def FR(r, k):
    """Calculate first ranking."""
    for i in range(k):
        if r[i] != 0:
            return i + 1
    return np.nan


def AR(r, k):
    """Calculate average ranking."""
    count = 0
    total = 0
    for i in range(k):
        if r[i] != 0:
            count = count + 1
            total = total + i + 1
    if total != 0:
        return total / count
    else:
        return np.nan


def MFR(r):
    """Calculate mean first ranking."""
    ret = [FR(r, i + 1) for i in range(len(r)) if r[i]]
    if len(ret) == 0:
        return np.nan
    return np.mean(ret)


def MAR(r):
    """Calculate mean first ranking."""
    ret = [AR(r, i + 1) for i in range(len(r)) if r[i]]
    if len(ret) == 0:
        return np.nan
    return np.mean(ret)


def get_r(pred, true, r_thresh=0.5, idx=0):
    """Sort predicted values based on output score."""
    zipped = list(zip(pred, true))
    zipped.sort(reverse=True, key=lambda x: x[idx])
    return [1 if i[0] > r_thresh and i[1] == 1 else 0 for i in zipped]


def rank_metr(pred, true, r_thresh=0.5, perfect=False):
    """Calculate all rank metrics."""
    if not any([i != 0 and i != 1 for i in pred]):
        print("Warning: Pred values are binary, not continuous.")
    ret = dict()
    kvals = [1, 3, 5, 10, 15, 20]
    r = get_r(pred, true, r_thresh, idx=1 if perfect else 0)
    last_vals = [0, 0, 0, 0]
    for k in kvals:
        if k > len(r):
            ret[f"nDCG@{k}"] = np.nan
            ret[f"MAP@{k}"] = np.nan
            ret[f"FR@{k}"] = np.nan
            ret[f"AR@{k}"] = np.nan
            continue
        ret[f"nDCG@{k}"] = ndcg_at_k(r, k)
        ret[f"MAP@{k}"] = mean_average_precision([r], k)
        ret[f"FR@{k}"] = FR(r, k)
        ret[f"AR@{k}"] = AR(r, k)
        last_vals = [ret[f"nDCG@{k}"], ret[f"MAP@{k}"], ret[f"FR@{k}"], ret[f"AR@{k}"]]

    mean_true = np.mean(true)
    if mean_true == 0 or mean_true == 1:
        ret["AUC"] = np.nan
    else:
        ret["AUC"] = roc_auc_score(true, pred)
    ret["MFR"] = MFR(r)
    ret["MAR"] = MAR(r)
    return ret
