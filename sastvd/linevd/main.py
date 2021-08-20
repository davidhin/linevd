import os

import dgl
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.codebert as cb
import sastvd.helpers.dataclasses as svddc
import sastvd.helpers.joern as svdj
import sastvd.ivdetect.evaluate as ivde
import torch as th
import torch.nn.functional as F
import torchmetrics
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv
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


def feature_extraction(_id):
    """Extract graph feature (basic)."""
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    n, e = ne_groupnodes(n, e)
    e = svdj.rdg(e, "pdg")
    n = svdj.drop_lone_nodes(n, e)

    # Map line numbers to indexing
    n = n.reset_index(drop=True).reset_index()
    iddict = pd.Series(n.index.values, index=n.id).to_dict()
    e.innode = e.innode.map(iddict)
    e.outnode = e.outnode.map(iddict)

    # Return plain-text code, line number list, innodes, outnodes
    return n.code.tolist(), n.id.tolist(), e.innode.tolist(), e.outnode.tolist()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, **kwargs):
        """Init."""
        super(BigVulDatasetLineVD, self).__init__(**kwargs)
        lines = ivde.get_dep_add_lines_bigvul()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = lines

    def item(self, _id, codebert=None):
        """Cache item."""
        savedir = svd.get_dir(svd.cache_dir() / "bigvul_linevd_codebert") / str(_id)
        if os.path.exists(savedir):
            g = load_graphs(str(savedir))[0][0]
            g.ndata["_VULN"] = g.ndata["_VULN"].long()
            return g
        code, linenum, ei, eo = feature_extraction(svddc.BigVulDataset.itempath(_id))
        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in linenum]
        else:
            vuln = [0 for _ in linenum]
        g = dgl.graph((eo, ei))
        code = [c.replace("\\t", "").replace("\\n", "") for c in code]
        features = [codebert.encode(c).detach().cpu() for c in chunks(code, 128)]
        g.ndata["_FEAT"] = th.cat(features)
        g.ndata["_LINE"] = th.Tensor(linenum).int()
        g.ndata["_VULN"] = th.Tensor(vuln).long()
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

    def __getitem__(self, idx):
        """Override getitem."""
        return self.item(self.idx2id[idx])


class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(self, batch_size: int = 32, sample: int = -1):
        """Init class from bigvul dataset."""
        super().__init__()
        self.train = BigVulDatasetLineVD(partition="train", sample=sample)
        self.val = BigVulDatasetLineVD(partition="val", sample=sample)
        self.test = BigVulDatasetLineVD(partition="test", sample=sample)
        codebert = cb.CodeBert()
        self.train.cache_items(codebert)
        self.val.cache_items(codebert)
        self.test.cache_items(codebert)
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return train dataloader."""
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return val dataloader."""
        return GraphDataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=self.batch_size)


# %%
class LitGAT(pl.LightningModule):
    """GAT."""

    def __init__(
        self,
        hfeat: int = 128,
        embfeat: int = 768,
        num_heads: int = 4,
        lr: float = 1e-3,
    ):
        """Initilisation."""
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.gat = GATConv(in_feats=embfeat, out_feats=hfeat, num_heads=num_heads)
        self.gat2 = GATConv(
            in_feats=hfeat * num_heads, out_feats=hfeat, num_heads=num_heads
        )
        self.gat3 = GATConv(
            in_feats=hfeat * num_heads, out_feats=hfeat, num_heads=num_heads
        )
        self.fc = th.nn.Linear(hfeat * num_heads, 2)
        self.accuracy = torchmetrics.Accuracy()
        self.auroc = torchmetrics.AUROC(compute_on_step=False)
        self.mcc = torchmetrics.MatthewsCorrcoef(2)

    def forward(self, g):
        """Forward pass."""
        h = self.gat(g, g.ndata["_FEAT"])
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)

        h = self.gat2(g, h)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)

        h = self.gat3(g, h)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)
        h = F.dropout(h, 0.3)

        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits = self(batch)
        labels = batch.ndata["_VULN"].long()
        loss = F.cross_entropy(logits, labels)

        pred = F.softmax(logits, dim=1)
        acc = self.accuracy(pred.argmax(1), labels)
        mcc = self.mcc(pred.argmax(1), labels)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, prog_bar=True, logger=True)
        self.log("train_mcc", mcc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        logits = self(batch)
        labels = batch.ndata["_VULN"].long()
        loss = F.cross_entropy(logits, labels)

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
        logits = self(batch)
        labels = batch.ndata["_VULN"].long()
        loss = F.cross_entropy(logits, labels)
        self.auroc.update(logits[:, 1], labels)
        self.log("test_auroc", self.auroc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return th.optim.AdamW(self.parameters(), lr=self.lr)


run_id = svd.get_run_id()
savepath = svd.get_dir(svd.processed_dir() / "gat" / run_id)
model = LitGAT()
data = BigVulDatasetLineVDDataModule(batch_size=32)
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
trainer = pl.Trainer(
    gpus=1,
    auto_lr_find=True,
    default_root_dir=savepath,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback],
)
tuned = trainer.tune(model, data)
trainer.fit(model, data)
