"""Main code for training. Probably needs refactoring."""
import os
from glob import glob

import dgl
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.codebert as cb
import sastvd.helpers.dclass as svddc
import sastvd.helpers.glove as svdg
import sastvd.helpers.joern as svdj
import sastvd.helpers.losses as svdloss
import sastvd.helpers.ml as ml
import sastvd.helpers.rank_eval as svdr
import sastvd.ivdetect.evaluate as ivde
import sastvd.linevd.gnnexplainer as lvdgne
import torch as th
import torch.nn.functional as F
import torchmetrics
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GraphConv
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from tqdm import tqdm


def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n[n.lineNumber != ""].copy()
    nl.lineNumber = nl.lineNumber.astype(int)
    nl = nl.sort_values(by="code", key=lambda x: x.str.len(), ascending=False)
    nl = nl.groupby("lineNumber").head(1)
    el = e.copy()
    el.innode = el.line_in
    el.outnode = el.line_out
    nl.id = nl.lineNumber
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False):
    """Extract graph feature (basic).

    _id = svddc.BigVulDataset.itempath(177775)
    _id = svddc.BigVulDataset.itempath(180189)
    _id = svddc.BigVulDataset.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evaluation).
    """
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    n, e = ne_groupnodes(n, e)

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0])
    n = svdj.drop_lone_nodes(n, e)

    # Plot graph
    # svdj.plot_graph_node_edge_df(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Map edge types
    etypes = e.etype.tolist()
    d = dict([(y, x) for x, y in enumerate(sorted(set(etypes)))])
    etypes = [d[i] for i in etypes]

    # Append function name to code
    if "+raw" not in graph_type:
        try:
            func_name = n[n.lineNumber == 1].name.item()
        except:
            print(_id)
            func_name = ""
        n.code = func_name + " " + n.name + " " + "</s>" + " " + n.code

    # Return plain-text code, line number list, innodes, outnodes
    return n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes


# %%
class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="cfgcdg", **kwargs):
        """Init."""
        super(BigVulDatasetLineVD, self).__init__(**kwargs)
        lines = ivde.get_dep_add_lines_bigvul()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = lines
        self.graph_type = gtype
        glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        self.glove_dict, _ = svdg.glove_dict(glove_path)

    def item(self, _id, codebert=None):
        """Cache item."""
        savedir = svd.get_dir(
            svd.cache_dir() / f"bigvul_linevd_codebert_{self.graph_type}"
        ) / str(_id)
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            if "_FVULN" not in g.ndata:
                g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
            if "_GLOVE" in g.ndata:  # TEMPORARY CHECK
                return g
        code, lineno, ei, eo, et = feature_extraction(
            svddc.BigVulDataset.itempath(_id), self.graph_type
        )
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
        g = dgl.graph((eo, ei))
        gembeds = th.Tensor(svdg.get_embeddings_list(code, self.glove_dict, 200))
        g.ndata["_GLOVE"] = gembeds
        if codebert:
            code = [c.replace("\\t", "").replace("\\n", "") for c in code]
            chunked_batches = svd.chunks(code, 128)
            features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
            g.ndata["_CODEBERT"] = th.cat(features)
        g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        g.ndata["_VULN"] = th.Tensor(vuln).float()
        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        g.edata["_ETYPE"] = th.Tensor(et).long()
        emb_path = svd.cache_dir() / f"codebert_method_level/{_id}.pt"
        g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g

    def cache_items(self, codebert):
        """Cache all items."""
        for i in tqdm(self.df.id.tolist()):
            try:
                self.item(i, codebert)
            except Exception as E:
                print(E)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert.

        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = svd.get_dir(svd.cache_dir() / "codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = svd.chunks((range(len(self.df))), 128)
        for idx_batch in tqdm(batches):
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            if set(batch_ids).issubset(done):
                continue
            texts = ["</s> " + ct for ct in batch_texts]
            embedded = codebert.encode(texts).detach().cpu()
            assert len(batch_texts) == len(batch_ids)
            for i in range(len(batch_texts)):
                th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")

    def __getitem__(self, idx):
        """Override getitem."""
        return self.item(self.idx2id[idx])


class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        dataargs = {"sample": sample, "gtype": gtype, "splits": splits}
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        codebert = cb.CodeBert()
        self.train.cache_codebert_method_level(codebert)
        self.val.cache_codebert_method_level(codebert)
        self.test.cache_codebert_method_level(codebert)
        self.train.cache_items(codebert)
        self.val.cache_items(codebert)
        self.test.cache_items(codebert)
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops

    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.NodeDataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=1,
        )

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val))))
            return self.node_dl(g)
        return GraphDataLoader(self.val, batch_size=self.batch_size)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=32)


# %%
class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512,
        embfeat: int = 768,
        num_heads: int = 4,
        lr: float = 1e-3,
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce",
        multitask: str = "linemethod",
        stmtweight: int = 5,
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()

        # Loss
        if self.hparams.loss == "sce":
            self.loss = svdloss.SCELoss(self.hparams.scea, 1 - self.hparams.scea)
            self.loss_f = th.nn.CrossEntropyLoss()
        else:
            self.loss = th.nn.CrossEntropyLoss(
                weight=th.Tensor([1, self.hparams.stmtweight]).cuda()
            )
            self.loss_f = th.nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC(compute_on_step=False)
        self.mcc = torchmetrics.MatthewsCorrcoef(2)

        # GraphConv Type
        hfeat = self.hparams.hfeat
        gatdrop = self.hparams.gatdropout
        numheads = self.hparams.num_heads
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}
        if self.hparams.gnntype == "gat":
            gnn = GATConv
            gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
            gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
            gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}

        # model: gat2layer
        if "gat" in self.hparams.model:
            self.gat = gnn(**gnn1_args)
            self.gat2 = gnn(**gnn2_args)
            fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
            self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
            self.fconly = th.nn.Linear(self.hparams.embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: mlp-only
        if "mlponly" in self.hparams.model:
            self.fconly = th.nn.Linear(self.hparams.embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        # model: contains femb
        if "+femb" in self.hparams.model:
            self.fc_femb = th.nn.Linear(self.hparams.embfeat * 2, self.hparams.hfeat)

        # self.resrgat = ResRGAT(hdim=768, rdim=1, numlayers=1, dropout=0)
        # self.gcn = GraphConv(embfeat, hfeat)
        # self.gcn2 = GraphConv(hfeat, hfeat)

        # Hidden Layers
        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass.

        data = BigVulDatasetLineVDDataModule(batch_size=1, sample=2, nsampling=True)
        g = next(iter(data.train_dataloader()))

        e_weights and h_override are just used for GNNExplainer.
        """
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata["_CODEBERT"]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                h = g.srcdata["_CODEBERT"]
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata["_CODEBERT"]
        else:
            g2 = g
            h = g.ndata["_CODEBERT"]
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
                    if out[0][idx].argmax() > 0:
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

    def plot_pr_curve(self):
        """Plot Precision-Recall Curve for Positive Class (after test)."""
        precision, recall, thresholds = precision_recall_curve(
            self.linevd_true, [i[1] for i in self.linevd_pred]
        )
        disp = PrecisionRecallDisplay(precision, recall)
        disp.plot()
        return

    def configure_optimizers(self):
        """Configure optimizer."""
        return th.optim.AdamW(self.parameters(), lr=self.lr)
