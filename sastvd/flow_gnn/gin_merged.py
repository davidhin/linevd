"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from code_gnn.globals import all_aggregate_functions
from code_gnn.models.base_module import BaseModule
from code_gnn.models.embedding_ids import node_type_map
from code_gnn.models.flow_gnn.ginconv import MyGINConv
from code_gnn.models.flow_gnn.mlp import MLP

import logging

logger = logging.getLogger(__name__)


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class FlowGNNLineVDModule:
    """
    FlowGNN
    """

    def __init__(
        self,
        input_dim,
        num_layers,
        num_mlp_layers,
        hidden_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        neighbor_pooling_type,
        **kwargs,
    ):
        super().__init__()

        self.loss_fn = BCELoss()

        self.save_hyperparameters()
        output_dim = 1
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # construct neural network layers

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                MyGINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        additional_element_size = 0
        if self.hparams.node_type_separate:
            additional_element_size = len(node_type_map)

        if self.hparams.label_style == "node":
            for layer in range(num_layers):
                i = hidden_dim
                o = hidden_dim
                if layer == 0:
                    i = hidden_dim + additional_element_size
                if layer == num_layers - 1:
                    o = output_dim
                self.linears_prediction.append(nn.Linear(i, o))
        else:
            for layer in range(num_layers):
                if layer == 0:
                    self.linears_prediction.append(
                        nn.Linear(input_dim + additional_element_size, output_dim)
                    )
                else:
                    self.linears_prediction.append(
                        nn.Linear(hidden_dim + additional_element_size, output_dim)
                    )

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("model arguments")
        # base
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=0)
        # FlowGNN
        parser.add_argument(
            "--num_layers", type=int, default=5, help="number of GIN layers to use"
        )
        parser.add_argument(
            "--num_mlp_layers",
            type=int,
            default=2,
            help="number of layers to use in each GIN layer's MLP",
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=32, help="width of the GIN hidden layers"
        )
        parser.add_argument(
            "--learn_eps",
            type=bool,
            default=False,
            help="whether or not to learn a weight for the epsilon value",
        )
        parser.add_argument(
            "--final_dropout",
            type=float,
            default=0.5,
            help="probability to use for the final dropout layer",
        )
        parser.add_argument(
            "--graph_pooling_type",
            type=str,
            default="sum",
            help="GIN graph pooling operator to use",
        )
        parser.add_argument(
            "--neighbor_pooling_type",
            type=str,
            default="sum",
            choices=all_aggregate_functions,
            help="GIN neighbor pooling operator to use",
        )
        parser.add_argument(
            "--node_type_separate",
            action="store_true",
            help="attach node type separately from data flow features",
        )
        return parent_parser

    def reset_parameters(self):
        """TODO"""
        pass

    def forward(self, g):
        h = g.ndata["h"]
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        if self.hparams.label_style == "node":
            # GIN paper page 3:
            # For node classification, the node representation h_v^K
            # of the final iteration is used for prediction.
            result = hidden_rep[-1]
            if self.hparams.node_type_separate:
                # if result.shape[0] != g.ndata['node_type'].shape[0]:
                #     logger.debug(f'{result.shape=} {result=}')
                #     logger.debug(f"{g.ndata['node_type'].shape=} {g.ndata['node_type']=}")
                result = torch.cat((result, g.ndata["node_type"]), dim=1)
            for fc in self.linears_prediction:
                result = fc(result)
            result = torch.sigmoid(result).squeeze(dim=-1)
        else:
            score_over_layer = torch.tensor(0)

            # perform pooling over all nodes in each graph in every layer
            for i, h in enumerate(hidden_rep):
                logger.warning("NOT WORKING, UNDER CONSTRUCTION...")
                # logger.debug(f'{h.shape=} {h=}')
                if self.hparams.node_type_separate:
                    # if h.shape[0] != g.ndata['node_type'].shape[0]:
                    #     logger.debug(f'{h.shape=} {h=}')
                    #     logger.debug(f"{g.ndata['node_type'].shape=} {g.ndata['node_type']=}")
                    h = torch.cat((h, g.ndata["node_type"]), dim=1)
                    # TODO: we want to pass this through a linear layer so that the one-hot gets picked up.
                h = self.pool(g, h)
                fc_out = self.linears_prediction[i](h)
                score_over_layer += self.drop(fc_out)

            result = torch.sigmoid(score_over_layer).squeeze(dim=-1)

        return result

    """
    base
    """

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def log_ranking_metrics(self, name, all_label, all_out, batch_num_nodes):
        all_graph_metrics = []
        prefix = 0
        for nn in batch_num_nodes:
            graph_out = all_out[prefix : prefix + nn]
            # graph_rank = [i for i, x in sorted(enumerate(graph_out), key=lambda x: x[1], reverse=True)]
            # # print(graph_rank)
            graph_label = all_label[prefix : prefix + nn]
            graph_metrics = rank_metr(graph_out, graph_label)
            all_graph_metrics.append(graph_metrics)
        keys = all_graph_metrics[0].keys()
        agg_metrics = {k: [] for k in keys}
        for d in all_graph_metrics:
            for k in keys:
                agg_metrics[k].append(d[k])
        mean_metrics = {k: np.average(v) for k, v in agg_metrics.items()}
        for k in keys:
            self.logger.experiment.add_scalar(
                name + "/rank/" + k.replace("@", "_"),
                mean_metrics[k],
                self.current_epoch,
            )

    def do_log(self, name, outputs):
        if any("loss" in o for o in outputs):
            all_loss = [o["loss"].item() for o in outputs]
            self.logger.experiment.add_scalar(
                name + "/avg_epoch_loss", np.average(all_loss), self.current_epoch
            )
            self.logger.experiment.add_scalar(
                name + "/total_epoch_loss", sum(all_loss), self.current_epoch
            )

        # collate outputs
        all_out = torch.cat([o["out"] for o in outputs]).float().cpu()
        all_pred = (all_out > 0.5).int().cpu()
        all_label = torch.cat([o["label"] for o in outputs]).int().cpu()
        batch_num_nodes = torch.cat([o["batch_num_nodes"] for o in outputs]).int().cpu()

        # classification metrics
        acc = accuracy_score(all_pred, all_label)
        prec = precision_score(all_pred, all_label, zero_division=0)
        rec = recall_score(all_pred, all_label, zero_division=0)
        f1 = f1_score(all_pred, all_label, zero_division=0)
        self.logger.experiment.add_scalar(name + "/class/acc", acc, self.current_epoch)
        self.logger.experiment.add_scalar(
            name + "/class/prec", prec, self.current_epoch
        )
        self.logger.experiment.add_scalar(name + "/class/rec", rec, self.current_epoch)
        self.logger.experiment.add_scalar(name + "/class/f1", f1, self.current_epoch)

        self.logger.experiment.add_scalar(
            name + "/meta/proportion_label", np.average(all_label), self.current_epoch
        )
        self.logger.experiment.add_scalar(
            name + "/meta/proportion_pred", np.average(all_pred), self.current_epoch
        )
        # self.logger.experiment.add_image(
        #     name + '/meta/predictions',
        #     np.expand_dims(np.stack((all_pred, all_label)), axis=0),
        #     self.current_epoch, dataformats='CHW'
        # )

        if any("loss_dim" in o for o in outputs):
            loss_percent = sum(o["loss_dim"] for o in outputs) / len(all_pred)
            self.logger.experiment.add_scalar(
                name + "/percent_sampled", loss_percent, self.current_epoch
            )

        # ranking metrics
        self.log_ranking_metrics(name, all_label, all_out, batch_num_nodes)

    def get_label(self, batch):
        if self.hparams.label_style == "node":
            label = batch.ndata["node_label"]
        elif self.hparams.label_style == "graph":
            graphs = dgl.unbatch(batch, batch.batch_num_nodes())
            label = torch.stack([g.ndata["graph_label"][0] for g in graphs])
        else:
            raise NotImplementedError(self.hparams.label_style)
        return label.float()

    def training_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        self.logger.experiment.add_scalar(
            "train/meta/original_label_proportion", torch.mean(label)
        )
        self.logger.experiment.add_scalar(
            "train/meta/original_label_len", label.shape[0]
        )
        # out_for_loss = out
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
        self.logger.experiment.add_scalar(
            "train/meta/undersampled_label_proportion", torch.mean(label)
        )
        self.logger.experiment.add_scalar(
            "train/meta/undersampled_label_len", label.shape[0]
        )
        loss = self.loss_fn(out, label)
        return {
            "loss": loss,
            "loss_dim": len(out),
            "out": out.detach(),
            "batch_num_nodes": batch.batch_num_nodes().detach(),
            "label": label,
        }

    def validation_step(self, batch, batch_idx):
        label = self.get_label(batch)
        # logger.info(f'val label {label.sum().item()}')
        out = self.forward(batch)
        pred = torch.gt(out, 0.5)
        self.log(
            "valid/f1",
            torch.tensor(
                f1_score(pred.int().cpu(), label.int().cpu(), zero_division=0)
            ),
            logger=False,
            batch_size=batch.batch_size,
        )
        self.log(
            "valid/acc",
            torch.tensor(accuracy_score(pred.int().cpu(), label.int().cpu())),
            logger=False,
            batch_size=batch.batch_size,
        )
        return {
            "out": out.detach(),
            "batch_num_nodes": batch.batch_num_nodes().detach(),
            "label": label,
        }

    def test_step(self, batch, batch_idx):
        label = self.get_label(batch)
        out = self.forward(batch)
        return {
            "out": out.detach(),
            "batch_num_nodes": batch.batch_num_nodes().detach(),
            "label": label,
        }

    def training_epoch_end(self, outputs):
        self.do_log("train", outputs)

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

        self.do_log("valid", outputs)

    def test_epoch_end(self, outputs):
        self.do_log("test", outputs)

    """
    linevd
    """

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass.

        data = BigVulDatasetLineVDDataModule(batch_size=1, sample=2, nsampling=True)
        g = next(iter(data.train_dataloader()))

        e_weights and h_override are just used for GNNExplainer.
        """
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                h = g.srcdata[self.EMBED]
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata[self.EMBED]
        else:
            g2 = g
            h = g.ndata[self.EMBED]
            if len(feat_override) > 0:
                h = g.ndata[feat_override]
            h_func = g.ndata["_FUNC_EMB"]
            hdst = h

        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device), th.rand(
                h_func.shape[0], 2
            ).to(self.device)

        # model: contains femb
        if "+femb" in self.hparams.model:
            h = th.cat([h, h_func], dim=1)
            h = F.elu(self.fc_femb(h))

        # Transform h_func if wrong size
        if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)

        # model: gat2layer
        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                h = self.gat(g, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
                h = self.gat2(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            elif "gat1layer" in self.hparams.model:
                h = self.gat(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            h = self.mlpdropout(F.elu(self.fc(h)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Edge masking (for GNNExplainer)
        if test and len(e_weights) > 0:
            g.ndata["h"] = h
            g.edata["ew"] = e_weights
            g.update_all(
                dgl.function.u_mul_e("h", "ew", "m"), dgl.function.mean("m", "h")
            )
            h = g.ndata["h"]

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            h = self.mlpdropout(F.elu(self.fconly(hdst)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        # Hidden layers
        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        h_func = self.fc2(
            h_func
        )  # Share weights between method-level and statement-level tasks

        if self.hparams.methodlevel:
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h"), None
        else:
            return h, h_func  # Return two values for multitask training

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        if self.hparams.methodlevel:
            if self.hparams.nsampling:
                raise ValueError("Cannot train on method level with nsampling.")
            labels = dgl.max_nodes(batch, "_VULN").long()
            labels_func = None
        else:
            if self.hparams.nsampling and not test:
                labels = batch[2][-1].dstdata["_VULN"].long()
                labels_func = batch[2][-1].dstdata["_FVULN"].long()
            else:
                labels = batch.ndata["_VULN"].long()
                labels_func = batch.ndata["_FVULN"].long()
        return logits, labels, labels_func

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels, labels_func = self.shared_step(
            batch
        )  # Labels func should be the method-level label for statements
        # print(logits.argmax(1), labels_func)
        loss1 = self.loss(logits[0], labels)
        if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)
        # Need some way of combining the losses for multitask training
        loss = 0
        if "line" in self.hparams.multitask:
            loss1 = self.loss(logits[0], labels)
            loss += loss1
        if "method" in self.hparams.multitask and not self.hparams.methodlevel:
            loss2 = self.loss(logits[1], labels_func)
            loss += loss2

        logits = logits[1] if self.hparams.multitask == "method" else logits[0]
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        if not self.hparams.methodlevel:
            acc_func = self.accuracy(logits.argmax(1), labels_func)
        mcc = self.mcc(pred.argmax(1), labels)
        # print(pred.argmax(1), labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        if not self.hparams.methodlevel:
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        logits, labels, labels_func = self.shared_step(batch)
        loss = 0
        if "line" in self.hparams.multitask:
            loss1 = self.loss(logits[0], labels)
            loss += loss1
        if "method" in self.hparams.multitask:
            loss2 = self.loss_f(logits[1], labels_func)
            loss += loss2

        logits = logits[1] if self.hparams.multitask == "method" else logits[0]
        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("val_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.auroc.update(logits[:, 1], labels)
        self.log("val_auroc", self.auroc, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        self.log("val_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels, _ = self.shared_step(
            batch, True
        )  # TODO: Make work for multitask

        if self.hparams.methodlevel:
            labels_f = labels
            return logits[0], labels_f, dgl.unbatch(batch)

        batch.ndata["pred"] = F.softmax(logits[0], dim=1)
        batch.ndata["pred_func"] = F.softmax(logits[1], dim=1)
        logits_f = []
        labels_f = []
        preds = []
        for i in dgl.unbatch(batch):
            preds.append(
                [
                    list(i.ndata["pred"].detach().cpu().numpy()),
                    list(i.ndata["_VULN"].detach().cpu().numpy()),
                    i.ndata["pred_func"].argmax(1).detach().cpu(),
                    list(i.ndata["_LINE"].detach().cpu().numpy()),
                ]
            )
            logits_f.append(dgl.mean_nodes(i, "pred_func").detach().cpu())
            labels_f.append(dgl.mean_nodes(i, "_FVULN").detach().cpu())
        return [logits[0], logits_f], [labels, labels_f], preds

    def test_epoch_end(self, outputs):
        """Calculate metrics for whole test set."""
        all_pred = th.empty((0, 2)).long().cuda()
        all_true = th.empty((0)).long().cuda()
        all_pred_f = []
        all_true_f = []
        all_funcs = []
        from importlib import reload

        reload(lvdgne)
        reload(ml)
        if self.hparams.methodlevel:
            for out in outputs:
                all_pred_f += out[0]
                all_true_f += out[1]
                for idx, g in enumerate(out[2]):
                    all_true = th.cat([all_true, g.ndata["_VULN"]])
                    gnnelogits = th.zeros((g.number_of_nodes(), 2), device="cuda")
                    gnnelogits[:, 0] = 1
                    if out[1][idx] == 1:
                        zeros = th.zeros(g.number_of_nodes(), device="cuda")
                        importance = th.ones(g.number_of_nodes(), device="cuda")
                        try:
                            if out[1][idx] == 1:
                                importance = lvdgne.get_node_importances(self, g)
                            importance = importance.unsqueeze(1)
                            gnnelogits = th.cat([zeros.unsqueeze(1), importance], dim=1)
                        except Exception as E:
                            print(E)
                            pass
                    all_pred = th.cat([all_pred, gnnelogits])
                    func_pred = out[0][idx].argmax().repeat(g.number_of_nodes())
                    all_funcs.append(
                        [
                            gnnelogits.detach().cpu().numpy(),
                            g.ndata["_VULN"].detach().cpu().numpy(),
                            func_pred.detach().cpu(),
                        ]
                    )
            all_true = all_true.long()
        else:
            for out in outputs:
                all_pred = th.cat([all_pred, out[0][0]])
                all_true = th.cat([all_true, out[1][0]])
                all_pred_f += out[0][1]
                all_true_f += out[1][1]
                all_funcs += out[2]
        all_pred = F.softmax(all_pred, dim=1)
        all_pred_f = F.softmax(th.stack(all_pred_f).squeeze(), dim=1)
        all_true_f = th.stack(all_true_f).squeeze().long()
        self.all_funcs = all_funcs
        self.all_true = all_true
        self.all_pred = all_pred
        self.all_pred_f = all_pred_f
        self.all_true_f = all_true_f

        # Custom ranked accuracy (inc negatives)
        self.res1 = ivde.eval_statements_list(all_funcs)

        # Custom ranked accuracy (only positives)
        self.res1vo = ivde.eval_statements_list(all_funcs, vo=True, thresh=0)

        # Regular metrics
        multitask_pred = []
        multitask_true = []
        for af in all_funcs:
            line_pred = list(zip(af[0], af[2]))
            multitask_pred += [list(i[0]) if i[1] == 1 else [1, 0] for i in line_pred]
            multitask_true += list(af[1])
        self.linevd_pred = multitask_pred
        self.linevd_true = multitask_true
        multitask_true = th.LongTensor(multitask_true)
        multitask_pred = th.Tensor(multitask_pred)
        self.f1thresh = ml.best_f1(multitask_true, [i[1] for i in multitask_pred])
        self.res2mt = ml.get_metrics_logits(multitask_true, multitask_pred)
        self.res2 = ml.get_metrics_logits(all_true, all_pred)
        self.res2f = ml.get_metrics_logits(all_true_f, all_pred_f)

        # Ranked metrics
        rank_metrs = []
        rank_metrs_vo = []
        for af in all_funcs:
            rank_metr_calc = svdr.rank_metr([i[1] for i in af[0]], af[1], 0)
            if max(af[1]) > 0:
                rank_metrs_vo.append(rank_metr_calc)
            rank_metrs.append(rank_metr_calc)
        try:
            self.res3 = ml.dict_mean(rank_metrs)
        except Exception as E:
            print(E)
            pass
        self.res3vo = ml.dict_mean(rank_metrs_vo)

        # Method level prediction from statement level
        method_level_pred = []
        method_level_true = []
        for af in all_funcs:
            method_level_true.append(1 if sum(af[1]) > 0 else 0)
            pred_method = 0
            for logit in af[0]:
                if logit[1] > 0.5:
                    pred_method = 1
                    break
            method_level_pred.append(pred_method)
        self.res4 = ml.get_metrics(method_level_true, method_level_pred)

        return
