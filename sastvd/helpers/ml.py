import pickle as pkl
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter


def get_metrics(true, pred, loss=-1, pr_auc=-1, roc_auc=-1):
    """Get relevant metrics given true labels and logits."""
    metrics = {}
    metrics["loss"] = loss
    metrics["acc"] = accuracy_score(true, pred)
    metrics["f1"] = f1_score(true, pred, zero_division=0)
    metrics["rec"] = recall_score(true, pred, zero_division=0)
    metrics["prec"] = precision_score(true, pred, zero_division=0)
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    metrics["fpr"] = -1
    metrics["fnr"] = -1
    if sum(true + pred) != 0:
        tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
        if fp + tn != 0:
            metrics["fpr"] = fp / (fp + tn)
        if fn + tp != 0:
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
        roc_auc = roc_auc_score(true, logits[:, 1])
    except:
        roc_auc = -1
    try:
        pr_auc = average_precision_score(true_oh, logits)
    except:
        pr_auc = -1
    ret = get_metrics(true, pred, loss, pr_auc, roc_auc)
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
        print("\x1b[40m\x1b[37m" + ret_str[:-1] + "\x1b[0m")
    return ret_str


def met_dict_to_writer(md, step, writer, prefix):
    """Given a dict of eval metrics, write to given Tensorboard writer."""
    for k, v in md.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def print_seperator(strings: list, max_len: int):
    """Print text inside a one-line string with "=" seperation to a max length.

    Args:
        strings (list): List of strings.
        max_len (int): Max length.
    """
    midpoints = int(max_len / len(strings))
    strings = [str(i) for i in strings]
    final_str = ""
    cutoff = max_len + (9 * len(strings))
    for s in strings:
        if "\x1b" in s:
            cutoff += 9
        len_s = len(s.replace("\x1b[32m", "").replace("\x1b[39m", ""))
        final_str += "\x1b[40m"
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += f" {s} "
        final_str += "=" * (int((midpoints / 2) - int(len_s / 2)) - 1)
        final_str += "\x1b[0m"
    print(final_str[:cutoff])


def dict_mean(dict_list):
    """Get mean of values from list of dicts.

    https://stackoverflow.com/questions/29027792
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict


class LogWriter:
    """Writer class for logging PyTorch model performance."""

    def __init__(
        self,
        model,
        path: str,
        max_patience: int = 100,
        log_every: int = 10,
        val_every: int = 50,
    ):
        """Init writer.

        Args:
            model: Pytorch model.
            path (str): Path to save log files.
        """
        self._model = model
        self._best_val_loss = 100
        self._patience = 0
        self._max_patience = max_patience
        self._epoch = 0
        self._step = 0
        self._path = Path(path)
        self._writer = SummaryWriter(path)
        self._log_every = log_every
        self._val_every = val_every
        self.save_attrs = ["_best_val_loss", "_patience", "_epoch", "_step"]

    def log(self, train_mets, val_mets):
        """Log information."""
        if self._step % self._log_every != 0:
            self.step()
            return

        if not self.log_val():
            met_dict_to_str(train_mets, "TR = ")
            met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
            self.step()
            return

        val_loss = val_mets["loss"]
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            with open(self._path / "best.model", "wb") as f:
                torch.save(self._model.state_dict(), f)
            best_model_string = "Best model saved: %.3f" % val_loss
            best_model_string = f"\x1b[32m{best_model_string}\x1b[39m"
            self._patience = 0
        else:
            self._patience += 1
            best_model_string = "No improvement."
        print_seperator(
            [
                f"Patience: {self._patience:03d}",
                f"Epoch: {self._epoch:03d}",
                f"Step: {self._step:03d}",
                best_model_string,
            ],
            131,
        )
        met_dict_to_str(train_mets, "TR = ")
        met_dict_to_writer(train_mets, self._step, self._writer, "TRN")
        met_dict_to_str(val_mets, "VA = ")
        met_dict_to_writer(val_mets, self._step, self._writer, "VAL")
        self.step()

    def test(self, test_mets):
        """Helper function to write test mets."""
        print_seperator(["\x1b[36mTest Set\x1b[39m"], 135)
        met_dict_to_str(test_mets, "TS = ")

    def log_val(self):
        """Check whether should validate or not."""
        if self._step % self._val_every == 0:
            return True
        return False

    def step(self):
        """Increment step."""
        self._step += 1

    def epoch(self):
        """Increment epoch."""
        self._epoch += 1

    def stop(self):
        """Check if should stop training."""
        return self._patience > self._max_patience

    def load_best_model(self):
        """Load best model."""
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "best.model"))

    def save_logger(self):
        """Save class attributes."""
        with open(self._path / "log.pkl", "wb") as f:
            f.write(pkl.dumps(dict([(i, getattr(self, i)) for i in self.save_attrs])))
        with open(self._path / "current.model", "wb") as f:
            torch.save(self._model.state_dict(), f)

    def load_logger(self):
        """Load class attributes."""
        with open(self._path / "log.pkl", "rb") as f:
            attrs = pkl.load(f)
            for k, v in attrs.items():
                setattr(self, k, v)
        torch.cuda.empty_cache()
        self._model.load_state_dict(torch.load(self._path / "current.model"))
