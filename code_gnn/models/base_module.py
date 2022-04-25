import logging
import random
from collections import defaultdict

import dgl
import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, average_precision_score, \
    ndcg_score

from code_gnn.models.rank_eval import rank_metr

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

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def log_ranking_metrics(self, name, all_label, all_out, batch_num_nodes):
        all_graph_metrics = []
        prefix = 0
        for nn in batch_num_nodes:
            graph_out = all_out[prefix:prefix + nn]
            # graph_rank = [i for i, x in sorted(enumerate(graph_out), key=lambda x: x[1], reverse=True)]
            # # print(graph_rank)
            graph_label = all_label[prefix:prefix + nn]
            graph_metrics = rank_metr(graph_out, graph_label)
            all_graph_metrics.append(graph_metrics)
        keys = all_graph_metrics[0].keys()
        agg_metrics = {k: [] for k in keys}
        for d in all_graph_metrics:
            for k in keys:
                agg_metrics[k].append(d[k])
        mean_metrics = {k: np.average(v) for k, v in agg_metrics.items()}
        for k in keys:
            self.logger.experiment.add_scalar(name + '/rank/' + k.replace("@", "_"), mean_metrics[k], self.current_epoch)

    def do_log(self, name, outputs):
        if any("loss" in o for o in outputs):
            all_loss = [o["loss"].item() for o in outputs]
            self.logger.experiment.add_scalar(name + '/avg_epoch_loss', np.average(all_loss), self.current_epoch)
            self.logger.experiment.add_scalar(name + '/total_epoch_loss', sum(all_loss), self.current_epoch)

        # collate outputs
        all_out = torch.cat([o["out"] for o in outputs]).float().cpu()
        all_pred = (all_out > 0.5).int().cpu()
        all_label = torch.cat([o["label"] for o in outputs]).int().cpu()
        batch_num_nodes = torch.cat([o["batch_num_nodes"] for o in outputs]).int().cpu()
        
        # breakpoint()

        # classification metrics
        acc = accuracy_score(all_pred, all_label)
        prec = precision_score(all_pred, all_label, zero_division=0)
        rec = recall_score(all_pred, all_label, zero_division=0)
        f1 = f1_score(all_pred, all_label, zero_division=0)
        self.logger.experiment.add_scalar(name + '/class/acc', acc, self.current_epoch)
        self.logger.experiment.add_scalar(name + '/class/prec', prec, self.current_epoch)
        self.logger.experiment.add_scalar(name + '/class/rec', rec, self.current_epoch)
        self.logger.experiment.add_scalar(name + '/class/f1', f1, self.current_epoch)

        self.logger.experiment.add_scalar(name + '/meta/proportion_label', np.average(all_label), self.current_epoch)
        self.logger.experiment.add_scalar(name + '/meta/proportion_pred', np.average(all_pred), self.current_epoch)
        # self.logger.experiment.add_image(
        #     name + '/meta/predictions',
        #     np.expand_dims(np.stack((all_pred, all_label)), axis=0),
        #     self.current_epoch, dataformats='CHW'
        # )

        if any("loss_dim" in o for o in outputs):
            loss_percent = sum(o["loss_dim"] for o in outputs) / len(all_pred)
            self.logger.experiment.add_scalar(name + '/percent_sampled', loss_percent, self.current_epoch)

        # ranking metrics
        self.log_ranking_metrics(name, all_label, all_out, batch_num_nodes)

    def get_label(self, batch):
        if self.hparams.label_style == 'node':
            label = batch.ndata[label_keys['node_label']]
        elif self.hparams.label_style == 'graph':
            graphs = dgl.unbatch(batch, batch.batch_num_nodes())
            label = torch.stack([g.ndata[label_keys['graph_label']][0] for g in graphs])
        else:
            raise NotImplementedError(self.hparams.label_style)
        return label.float()

    def training_step(self, batch, batch_idx):
        # batch, batch_label = batch
        label = self.get_label(batch)
        out = self.forward(batch)
        # breakpoint()
        self.logger.experiment.add_scalar('train/meta/original_label_proportion', torch.mean(label))
        self.logger.experiment.add_scalar('train/meta/original_label_len', label.shape[0])
        # out_for_loss = out
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
        self.logger.experiment.add_scalar('train/meta/undersampled_label_proportion', torch.mean(label))
        self.logger.experiment.add_scalar('train/meta/undersampled_label_len', label.shape[0])
        loss = self.loss_fn(out, label)
        return {
            "loss": loss, "loss_dim": len(out),
            "out": out.detach(),
            "batch_num_nodes": batch.batch_num_nodes().detach(),
            "label": label,
        }

    def validation_step(self, batch, batch_idx):
        # breakpoint()
        # batch, batch_label = batch
        label = self.get_label(batch)
        # logger.info(f'val label {label.sum().item()}')
        out = self.forward(batch)
        pred = torch.gt(out, 0.5)
        self.log('valid/f1', torch.tensor(f1_score(pred.int().cpu(), label.int().cpu(), zero_division=0)), logger=False,
                 batch_size=batch.batch_size)
        self.log('valid/acc', torch.tensor(accuracy_score(pred.int().cpu(), label.int().cpu())), logger=False,
                 batch_size=batch.batch_size)
        return {
            "out": out.detach(),
            "batch_num_nodes": batch.batch_num_nodes().detach(),
            "label": label
        }

    def test_step(self, batch, batch_idx):
        # batch, batch_label = batch
        label = self.get_label(batch)
        out = self.forward(batch)
        return {
            "out": out.detach(),
            "batch_num_nodes": batch.batch_num_nodes().detach(),
            "label": label
        }

    def training_epoch_end(self, outputs):
        self.do_log('train', outputs)

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

        self.do_log('valid', outputs)

    def test_epoch_end(self, outputs):
        self.do_log('test', outputs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BaseModule arguments")
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        return parent_parser