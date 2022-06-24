import logging
import random

import dgl
import pytorch_lightning as pl
from sklearn.metrics import roc_curve
import torch
import torchmetrics
from matplotlib import pyplot as plt

import torch
from torch.nn import BCELoss

logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

label_keys = {
    "graph_label": "_FVULN",
    "node_label": "_VULN",
}


class BaseModule(pl.LightningModule):
    def __init__(
        self,
        undersample_factor=None,
        test_every=False,
        ):
        super().__init__()
        self.class_threshold = 0.5

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.Precision(),
            torchmetrics.Recall(),
            torchmetrics.F1Score(),
            ])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        if test_every:
            self.test_every_metrics = metrics.clone(prefix='test_every_')
        else:
            self.test_every_metrics = None
        self.test_metrics = metrics.clone(prefix='test_')
        
        self.loss_fn = BCELoss()

    def get_label(self, batch):
        if self.hparams.label_style == "node":
            label = batch.ndata[label_keys["node_label"]]
        elif self.hparams.label_style == "graph":
            graphs = dgl.unbatch(batch, batch.batch_num_nodes())
            label = torch.stack([g.ndata[label_keys["graph_label"]][0] for g in graphs])
        else:
            raise NotImplementedError(self.hparams.label_style)
        return label.float()

    def resample(self, batch, out, label):
        """Resample logits and labels to balance vuln/nonvuln classes"""
        self.log(
            "meta/train_original_label_proportion",
            torch.mean(label).float().item(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        self.log(
            "meta/train_original_label_len",
            torch.tensor(label.shape[0]).float(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        vuln_indices = label.nonzero().squeeze().tolist()
        num_indices_to_sample = round(
            len(vuln_indices) * self.hparams.undersample_factor
        )
        nonvuln_indices = (label == 0).nonzero().squeeze().tolist()
        nonvuln_indices = random.sample(nonvuln_indices, num_indices_to_sample)
        # TODO: Does this need to be sorted?
        indices = vuln_indices + nonvuln_indices
        out = out[indices]
        label = label[indices]
        self.log(
            "meta/train_resampled_label_proportion",
            torch.mean(label).item(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        self.log(
            "meta/train_resampled_label_len",
            torch.tensor(label.shape[0]).float(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        return out, label
    

    def log_loss(self, name, loss, batch):
        self.log(
            f"{name}_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=batch.batch_size,
        )


    def training_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)
        
        if self.hparams.label_style == "node" and self.hparams.undersample_factor is not None:
            self.resample(self, batch, out, label)
        self.log_loss("train", loss, batch)
        output = self.train_metrics(out, label.int())
        self.log_dict(output, batch_size=batch.batch_size)

        return loss


    def on_after_backward(self):
        """https://github.com/Lightning-AI/lightning/issues/2660#issuecomment-699020383"""
        if self.global_step % 100 == 0:  # don't make the tf file huge
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)

        if dataloader_idx == 0:  # val set
            self.log_loss("val", loss, batch)
            output = self.val_metrics(out, label.int())
            self.log_dict(output, batch_size=batch.batch_size)
        elif dataloader_idx == 1:  # test set (--test_every)
            self.log_loss("test_every", loss, batch)
            output = self.test_every_metrics(out, label.int())
            self.log_dict(output, batch_size=batch.batch_size)


    def test_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)

        self.log_loss("test", loss, batch)
        output = self.test_metrics(out, label.int())
        self.log_dict(output, batch_size=batch.batch_size)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModule arguments")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        return parent_parser
