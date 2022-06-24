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
from torch.optim import Adam

from sastvd.linevd.datamodule import BigVulDatasetLineVDDataModule

logger = logging.getLogger(__name__)

logging.getLogger("matplotlib").setLevel(logging.WARNING)

label_keys = {
    "graph_label": "_FVULN",
    "node_label": "_VULN",
}


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
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.use_lr_scheduler is not None:
            # if self.hparams.use_lr_scheduler == "OneCycleLR":
            #     lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #         optimizer,
            #         max_lr=self.hparams.learning_rate,
            #         steps_per_epoch=self.hparams.steps_per_epoch,
            #         epochs=self.hparams.max_epochs,
            #         anneal_strategy="linear",
            #     )
            if self.hparams.use_lr_scheduler == "MultiplicativeLR":
                split = self.hparams.use_lr_scheduler.split("_")
                if len(split) == 2:
                    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda x: float(split[1]))
                else:
                    lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda x: 0.95)
            elif self.hparams.use_lr_scheduler == "ExponentialLR":
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def get_label(self, batch):
        if self.hparams.label_style == "node":
            label = batch.ndata[label_keys["node_label"]]
        elif self.hparams.label_style == "graph":
            graphs = dgl.unbatch(batch, batch.batch_num_nodes())
            label = torch.stack([g.ndata[label_keys["graph_label"]][0] for g in graphs])
        else:
            raise NotImplementedError(self.hparams.label_style)
        return label.float()

    def training_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        self.log(
            "train_meta_original_label_proportion",
            torch.mean(label).item(),
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        self.log(
            "train_meta_original_label_len",
            label.shape[0],
            on_step=True,
            on_epoch=False,
            batch_size=batch.batch_size,
        )
        if self.hparams.label_style == "node":
            if self.hparams.undersample_factor is not None:
                vuln_indices = label.nonzero().squeeze().tolist()
                num_indices_to_sample = round(
                    len(vuln_indices) * self.hparams.undersample_factor
                )
                nonvuln_indices = (label == 0).nonzero().squeeze().tolist()
                nonvuln_indices = random.sample(nonvuln_indices, num_indices_to_sample)
                # unif = -label_for_loss + 1
                # nonvuln_indices = unif.float().multinomial(num_indices_to_sample)
                # TODO: Does this need to be sorted?
                indices = vuln_indices + nonvuln_indices
                out = out[indices]
                label = label[indices]
                self.log(
                    "train_meta_resampled_label_proportion",
                    torch.mean(label).item(),
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch.batch_size,
                )
                self.log(
                    "train_meta_resampled_label_len",
                    label.shape[0],
                    on_step=True,
                    on_epoch=False,
                    batch_size=batch.batch_size,
                )
        loss = self.loss_fn(out, label)
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        # self.log_class_metrics("train", out, label)
        out = out.detach().float()
        label = label.detach().int()
        self.train_accuracy(out, label)
        self.train_precision(out, label)
        self.train_recall(out, label)
        self.train_f1(out, label)
        self.log(
            "train_class_accuracy", self.train_accuracy, on_step=True, on_epoch=True, batch_size=batch.batch_size,
        )
        self.log(
            "train_class_precision", self.train_precision, on_step=True, on_epoch=True, batch_size=batch.batch_size,
        )
        self.log("train_class_recall", self.train_recall, on_step=True, on_epoch=True, batch_size=batch.batch_size,)
        self.log("train_class_f1", self.train_f1, on_step=True, on_epoch=True, batch_size=batch.batch_size,)

        return loss

    def on_after_backward(self):
        """https://github.com/Lightning-AI/lightning/issues/2660#issuecomment-699020383"""
        if self.global_step % 100 == 0:  # don't make the tf file huge
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)
        # self.log_class_metrics("val", out, label)

        out = out.detach().float()
        label = label.detach().int()
        if dataloader_idx == 0:
            self.log(
                "val_loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                batch_size=batch.batch_size,
            )
            self.val_accuracy(out, label)
            self.val_precision(out, label)
            self.val_recall(out, label)
            self.val_f1(out, label)
            self.log("val_class_accuracy", self.val_accuracy, on_step=True, on_epoch=True, batch_size=batch.batch_size,)
            self.log(
                "val_class_precision", self.val_precision, on_step=True, on_epoch=True, batch_size=batch.batch_size,
            )
            self.log("val_class_recall", self.val_recall, on_step=True, on_epoch=True, batch_size=batch.batch_size,)
            self.log("val_class_f1", self.val_f1, on_step=True, on_epoch=True, batch_size=batch.batch_size,)
        elif dataloader_idx == 1:
            self.log(
                "test_loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                batch_size=batch.batch_size,
            )
            self.test_accuracy(out, label)
            self.test_precision(out, label)
            self.test_recall(out, label)
            self.test_f1(out, label)
            self.log(
                f"test_class_accuracy", self.test_accuracy, on_step=True, on_epoch=True, batch_size=batch.batch_size,
            )
            self.log(
                f"test_class_precision", self.test_precision, on_step=True, on_epoch=True, batch_size=batch.batch_size,
            )
            self.log(f"test_class_recall", self.test_recall, on_step=True, on_epoch=True, batch_size=batch.batch_size,)
            self.log(f"test_class_f1", self.test_f1, on_step=True, on_epoch=True, batch_size=batch.batch_size,)


    def test_step(self, batch, batch_idx):
        # breakpoint()
        label = self.get_label(batch)
        out = self.forward(batch)
        loss = self.loss_fn(out, label)
        self.log(
            "test_loss", loss.item(), on_step=True, on_epoch=True, batch_size=batch.batch_size
        )
        # self.log_class_metrics("test", out, label)
        out = out.detach().float()
        label = label.detach().int()
        self.test_accuracy(out, label)
        self.test_precision(out, label)
        self.test_recall(out, label)
        self.test_f1(out, label)
        self.log(
            f"test_class_accuracy", self.test_accuracy, on_step=True, on_epoch=True, batch_size=batch.batch_size,
        )
        self.log(
            f"test_class_precision", self.test_precision, on_step=True, on_epoch=True, batch_size=batch.batch_size,
        )
        self.log(f"test_class_recall", self.test_recall, on_step=True, on_epoch=True, batch_size=batch.batch_size,)
        self.log(f"test_class_f1", self.test_f1, on_step=True, on_epoch=True, batch_size=batch.batch_size,)

    def training_epoch_end(self, outputs):
        self.log("epoch", self.current_epoch)

    def validation_epoch_end(self, outputs):
        if self.hparams.roc_every is not None and (
            self.current_epoch == 0
            or (
                self.current_epoch != 1
                and ((self.current_epoch - 1) % self.hparams.roc_every == 0)
            )
        ):
            all_label = torch.cat([o["label"] for o in outputs]).cpu().numpy()
            all_out = torch.cat([o["out"] for o in outputs]).cpu().numpy()
            fpr, tpr, thresholds = roc_curve(all_label, all_out)
            logger.info(f"ROC curve thresholds {thresholds}")
            plt.close()
            plt.plot(fpr, tpr, marker=".", label="ROC")
            plt.plot([0, 1], [0, 1], linestyle="--", label="Baseline")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.title(f"ROC curve epoch {self.current_epoch}")
            filename = f"img_{self.current_epoch}.png"
            logger.info(f"Log ROC curve to {filename}")
            plt.savefig(filename)
            plt.show()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModule arguments")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        return parent_parser
