import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)


def get_metrics(true, pred, loss=-1, pr_auc=-1):
    """Get relevant metrics given true labels and logits."""
    metrics = {}
    metrics["loss"] = loss
    metrics["f1"] = f1_score(true, pred, zero_division=0)
    metrics["rec"] = recall_score(true, pred)
    metrics["prec"] = precision_score(true, pred, zero_division=0)
    try:
        metrics["roc_auc"] = roc_auc_score(true, pred)
    except:
        metrics["roc_auc"] = 0
    metrics["pr_auc"] = pr_auc
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    metrics["fpr"] = fp / (fp + tn)
    metrics["fnr"] = fn / (fn + tp)
    return metrics


def get_metrics_logits(true, logits):
    """Call get_metrics with logits."""
    loss = F.cross_entropy(logits, true).detach().cpu().item()
    if torch.is_tensor(true):
        true_oh = torch.nn.functional.one_hot(true).detach().cpu().numpy()
        true = true.detach().cpu().numpy()
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    pred = logits.argmax(1)
    try:
        pr_auc = average_precision_score(true_oh, logits)
    except:
        pr_auc = -1
    ret = get_metrics(true, pred, loss, pr_auc)
    return ret


def met_dict_to_str(md, prefix="", verbose=1):
    """Convert metric dictionary to string for printing."""
    ret_str = prefix
    for k, v in md.items():
        if k == "loss":
            ret_str += k + ": " + "%.5f" % v + " | "
        else:
            ret_str += k + ": " + "%.3f" % v + " | "
    if verbose > 0:
        print(ret_str)
    return ret_str


def met_dict_to_writer(md, step, writer, prefix):
    """Given a dict of eval metrics, write to given Tensorboard writer."""
    for k, v in md.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def print_seperator(patience, step, best_model_string):
    """Print during training helper function."""
    print(
        "\x1b[40m"
        + "=" * 15
        + f" Patience {patience:03d} "
        + "=" * 15
        + f" Step {step:03d} "
        + "=" * 15
        + f"{best_model_string}\x1b[40m"
        + "=" * 15
        + "\x1b[0m"
    )
