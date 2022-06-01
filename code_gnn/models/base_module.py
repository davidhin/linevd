import logging
import random
from collections import defaultdict

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from matplotlib import pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, average_precision_score, \
#     ndcg_score

from code_gnn.models.rank_eval import rank_metr
from sastvd import print_memory_mb

try:
    from torchmetrics import F1Score
except ImportError:
    F1Score = torchmetrics.F1
from torch.nn import BCELoss
from torch.optim import Adam

logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

# label_keys = {
#     "graph_label": "graph_label",
#     "node_label": "node_label",
# }
label_keys = {
    "graph_label": "_FVULN",
    "node_label": "_VULN",
}


class DistributionTracker:
    def __init__(self):
        self.predictions = defaultdict(int)

    def __call__(self, *args, **kwargs):
        predictions = args[0]
        for prediction in predictions:
            self.predictions[prediction] += 1


from torchmetrics import Metric


class BinaryPredictionDistribution(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("zeros", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ones", default=torch.tensor(0), dist_reduce_fx="sum")

    def _update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == 1)
        self.total += torch.sum(preds == 0)

    def _compute(self):
        return self.zeros.float() / self.ones


class BaseModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_fn = BCELoss()
        self.class_threshold = 0.5

        self.train_accuracy = torchmetrics.Accuracy()
        self.train_precision = torchmetrics.Precision()
        self.train_recall = torchmetrics.Recall()
        self.train_f1 = torchmetrics.F1Score()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_precision = torchmetrics.Precision()
        self.val_recall = torchmetrics.Recall()
        self.val_f1 = torchmetrics.F1Score()
        self.test_accuracy = torchmetrics.Accuracy()
        self.test_precision = torchmetrics.Precision()
        self.test_recall = torchmetrics.Recall()
        self.test_f1 = torchmetrics.F1Score()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def get_label(self, batch):
        if self.hparams.label_style == 'node':
            label = batch.ndata[label_keys['node_label']]
        elif self.hparams.label_style == 'graph':
            graphs = dgl.unbatch(batch, batch.batch_num_nodes())
            label = torch.stack([g.ndata[label_keys['graph_label']][0] for g in graphs])
        else:
            raise NotImplementedError(self.hparams.label_style)
        return label.float()

    # def log_class_metrics(self, name, out, label):
    #     for m_name, m in self.metrics[name].items():
    #         m(out.float().to(self.device), label.int().to(self.device))
    #         self.log(f'{name}/class/{m_name}', m, on_step=True, on_epoch=True)

    def training_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        self.log(f"train/meta/original_label_proportion", torch.mean(label), on_step=True, on_epoch=False, batch_size=batch.batch_size)
        self.log(f"train/meta/original_label_len", label.shape[0], on_step=True, on_epoch=False, batch_size=batch.batch_size)
        if self.hparams.label_style == 'node':
            if self.hparams.undersample_factor is not None:
                vuln_indices = label.nonzero().squeeze().tolist()
                num_indices_to_sample = round(len(vuln_indices) * self.hparams.undersample_factor)
                nonvuln_indices = (label == 0).nonzero().squeeze().tolist()
                nonvuln_indices = random.sample(nonvuln_indices, num_indices_to_sample)
                # unif = -label_for_loss + 1
                # nonvuln_indices = unif.float().multinomial(num_indices_to_sample)
                # TODO: Does this need to be sorted?
                indices = vuln_indices + nonvuln_indices
                out = out[indices]
                label = label[indices]
                self.log(f"train/meta/resampled_label_proportion", torch.mean(label), on_step=True, on_epoch=False, batch_size=batch.batch_size)
                self.log(f"train/meta/resampled_label_len", label.shape[0], on_step=True, on_epoch=False, batch_size=batch.batch_size)
        loss = self.loss_fn(out, label)
        self.log(f'train/loss', loss, on_step=True, on_epoch=True, batch_size=batch.batch_size)
        # self.log_class_metrics("train", out, label)
        self.train_accuracy(out, label.int())
        self.train_precision(out, label.int())
        self.train_recall(out, label.int())
        self.train_f1(out, label.int())
        self.log(f'train/class/accuracy', self.train_accuracy, on_step=True, on_epoch=True)
        self.log(f'train/class/precision', self.train_precision, on_step=True, on_epoch=True)
        self.log(f'train/class/recall', self.train_recall, on_step=True, on_epoch=True)
        self.log(f'train/class/f1', self.train_f1, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)
        self.log(f'valid/loss', loss, on_step=True, on_epoch=True, batch_size=batch.batch_size)
        # self.log_class_metrics("val", out, label)
        self.val_accuracy(out, label.int())
        self.val_precision(out, label.int())
        self.val_recall(out, label.int())
        self.val_f1(out, label.int())
        self.log(f'val/class/accuracy', self.val_accuracy, on_step=True, on_epoch=True)
        self.log(f'val/class/precision', self.val_precision, on_step=True, on_epoch=True)
        self.log(f'val/class/recall', self.val_recall, on_step=True, on_epoch=True)
        self.log(f'val/class/f1', self.val_f1, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)
        self.log(f'test/loss', loss, on_step=True, on_epoch=True, batch_size=batch.batch_size)
        # self.log_class_metrics("test", out, label)
        self.test_accuracy(out, label.int())
        self.test_precision(out, label.int())
        self.test_recall(out, label.int())
        self.test_f1(out, label.int())
        self.log(f'test/class/accuracy', self.test_accuracy, on_step=True, on_epoch=True)
        self.log(f'test/class/precision', self.test_precision, on_step=True, on_epoch=True)
        self.log(f'test/class/recall', self.test_recall, on_step=True, on_epoch=True)
        self.log(f'test/class/f1', self.test_f1, on_step=True, on_epoch=True)

    def training_epoch_end(self, outputs):
        self.log('epoch', self.current_epoch)

    def validation_epoch_end(self, outputs):
        if self.hparams.roc_every is not None and (
                self.current_epoch == 0 or
                (self.current_epoch != 1 and ((self.current_epoch - 1) % self.hparams.roc_every == 0))
        ):
            all_label = torch.cat([o["label"] for o in outputs]).cpu().numpy()
            all_out = torch.cat([o["out"] for o in outputs]).cpu().numpy()
            fpr, tpr, thresholds = roc_curve(all_label, all_out)
            logger.info(f'ROC curve thresholds {thresholds}')
            plt.close()
            plt.plot(fpr, tpr, marker='.', label='ROC')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Baseline')
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.title(f'ROC curve epoch {self.current_epoch}')
            filename = f'img_{self.current_epoch}.png'
            logger.info(f'Log ROC curve to {filename}')
            plt.savefig(filename)
            plt.show()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModule arguments")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        return parent_parser