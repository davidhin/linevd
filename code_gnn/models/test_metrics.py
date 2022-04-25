import numpy as np

from code_gnn.models.base_module import ranking_metrics


def test_sanity():
    all_label = np.array([0, 1, 0, 0, 1])
    all_out = np.array([0.1, 0.8, 0.1, 0.7, 0.6])
    metrics = ranking_metrics(all_label, all_out, [2, 3])
    print(metrics)


def test_mrr():
    """https://en.wikipedia.org/wiki/Mean_reciprocal_rank"""
    all_label = np.array([0, 0, 1] + [0, 1, 0] + [1, 0, 0])
    all_out = np.array([0.3, 0.2, 0.1] + [0.3, 0.2, 0.1] + [0.3, 0.2, 0.1])
    metrics = ranking_metrics(all_label, all_out, [3, 3, 3])
    assert metrics[3] - 0.6111 < 0.0001


def test_mfr():
    """https://en.wikipedia.org/wiki/Mean_reciprocal_rank"""
    all_label = np.array([0, 0, 1] + [0, 1, 0] + [1, 0, 0])
    all_out = np.array([0.3, 0.2, 0.1] + [0.3, 0.2, 0.1] + [0.3, 0.2, 0.1])
    metrics = ranking_metrics(all_label, all_out, [3, 3, 3])
    assert metrics[2] == 2.0


def test_map5():
    """https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52"""
    all_label = np.array([1, 0, 0, 1, 1])
    all_out = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    metrics = ranking_metrics(all_label, all_out, [5])
    assert metrics[1] - 0.7 < 0.0001

    all_label = np.array([1, 1, 1, 0, 0])
    all_out = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    metrics = ranking_metrics(all_label, all_out, [5])
    assert metrics[1] - 1.0 < 0.0001
