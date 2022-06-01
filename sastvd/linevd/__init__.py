"""Main code for training. Probably needs refactoring."""
import os
from glob import glob

import dgl
import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.helpers.joern as svdj
import sastvd.ivdetect.evaluate as ivde
import torch as th
from dgl.data.utils import load_graphs, save_graphs
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm



def ne_groupnodes(n, e):
    """Group nodes with same line number."""
    nl = n.copy()
    el = e.copy()
    nl = nl.dropna(subset=["lineNumber"])
    nl = nl.groupby("lineNumber").head(1)
    nl["nodeId"] = nl.id
    nl.id = nl.lineNumber
    el.innode = el.line_in
    el.outnode = el.line_out
    nl = svdj.drop_lone_nodes(nl, el)
    el = el.drop_duplicates(subset=["innode", "outnode", "etype"])
    el = el[el.innode.apply(lambda x: isinstance(x, float))]
    el = el[el.outnode.apply(lambda x: isinstance(x, float))]
    el.innode = el.innode.astype(int)
    el.outnode = el.outnode.astype(int)
    return nl, el


def feature_extraction(_id, graph_type="cfgcdg", return_nodes=False, return_node_ids=False, return_iddict=False, group=True, return_node_types=False):
    """Extract graph feature (basic).

    _id = svddc.svdds.itempath(177775)
    _id = svddc.svdds.itempath(180189)
    _id = svddc.svdds.itempath(178958)

    return_nodes arg is used to get the node information (for empirical evalu
    ation).
    """
    # Get CPG
    n, e = svdj.get_node_edges(_id)
    if group:
        n, e = svd.ne_groupnodes(n, e)
    else:
        n = n[n.lineNumber != ""].copy()
        n.lineNumber = n.lineNumber.astype(int)
        n["nodeId"] = n.id
        e.innode = e.innode.astype(int)
        e.outnode = e.outnode.astype(int)
        n = svdj.drop_lone_nodes(n, e)
        e = e.drop_duplicates(subset=["innode", "outnode", "etype"])

    # Return node metadata
    if return_nodes:
        return n

    # Filter nodes
    e = svdj.rdg(e, graph_type.split("+")[0])
    n = svdj.drop_lone_nodes(n, e)

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
        pass
    else:
        n.code = "</s>" + " " + n.code

    ret = [n.code.tolist(), n.lineNumber.tolist(), e.innode.tolist(), e.outnode.tolist(), etypes]
    if return_node_ids:
        ret.append(n.nodeId.tolist())
    if return_node_types:
        label = n._label.tolist()
        # TODO: replace calls to <operator>.assignment and such with name attribute
        ret.append(label)
    if return_iddict:
        ret.append(iddict)
    # Return plain-text code, line number list, innodes, outnodes
    return tuple(ret)


# %%
class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="pdg", feat="all", cache_all=False, **kwargs):
        """Init."""
        self.graph_type = gtype
        self.feat = feat
        super(BigVulDatasetLineVD, self).__init__(feat=feat, **kwargs)
        lines = ivde.get_dep_add_lines_bigvul()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.cache_all = cache_all
        self.cache_all_cache = {}
        self.lines = lines

    def item(self, _id, codebert=None, must_load=False):
        """Cache item."""

        if self.cache_all and not must_load:
            if _id in self.cache_all_cache:
                # print("return from cache")
                return self.cache_all_cache[_id]
            else:
                # print("load into cache")
                g = self.item(_id, codebert, must_load=True)
                self.cache_all_cache[_id] = g
                return g

        savedir = svd.get_dir(
            svd.cache_dir() / f"bigvul_linevd_codebert_{self.graph_type}_{self.feat}"
        ) / str(_id)
        if os.path.exists(savedir):
            try:
                g = load_graphs(str(savedir))[0][0]
            except Exception:
                savedir.unlink()
                return self.item(_id, codebert=codebert, must_load=must_load)
            if "_SASTRATS" in g.ndata:
                g.ndata.pop("_SASTRATS")
                g.ndata.pop("_SASTCPP")
                g.ndata.pop("_SASTFF")
            if "_DATAFLOW" in g.ndata:
                g.ndata.pop("_DATAFLOW")
            if "_CODEBERT" in g.ndata:
                if self.feat == "codebert":
                    for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                if self.feat == "glove":
                    for i in ["_CODEBERT", "_DOC2VEC", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
                if self.feat == "doc2vec":
                    for i in ["_CODEBERT", "_GLOVE", "_RANDFEAT"]:
                        g.ndata.pop(i, None)
            # print("load from file")
            return g
        code, lineno, ei, eo, et, nids, ntypes, iddict = feature_extraction(
            svddc.svdds.itempath(_id), self.graph_type, return_node_ids=True, return_iddict=True, group=False, return_node_types=True,
        )

        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
        g = dgl.graph((eo, ei))
        g.ndata["_LINE"] = th.Tensor(lineno).int()
        g.ndata["_VULN"] = th.Tensor(vuln).float()

        # Get dataflow features
        if "_1G_DATAFLOW" in self.feat:
            def get_dataflow_1g_features(_id):
                dgl_feat = th.zeros((g.number_of_nodes(), self.df_1g_max_idx*2))

                nids_to_1g_df = self.df_1g.groupby("graph_id").get_group(_id)
                node_id_dgl = nids_to_1g_df["node_id"].map(iddict)
                # all_nids = [i for i in node_id_dgl.dropna().astype(int).tolist() if i in iddict.values()]
                all_nids = node_id_dgl.dropna().astype(int).sort_values().unique().tolist()
                nids_gen = dict(zip(node_id_dgl, nids_to_1g_df["gen"]))
                nids_kill = dict(zip(node_id_dgl, nids_to_1g_df["kill"]))
                for nid in range(len(dgl_feat)):
                    gen_f = nids_gen.get(nid, "")
                    kill_f = nids_kill.get(nid, "")
                    for i in [int(s) for s in sorted(gen_f.split(",")) if s.isdigit()]:
                        if i in iddict and iddict[i] in all_nids:
                            dgl_feat[nid, all_nids.index(iddict[i])] = 1
                    for i in [int(s) for s in sorted(kill_f.split(",")) if s.isdigit()]:
                        if i in iddict and iddict[i] in all_nids:
                            dgl_feat[nid, self.df_1g_max_idx+all_nids.index(iddict[i])] = 1
                return dgl_feat
            g.ndata["_1G_DATAFLOW"] = get_dataflow_1g_features(_id)
        
        if "_ABS_DATAFLOW" in self.feat:
            def get_abs_dataflow_features(_id):
                dgl_feat = th.zeros((g.number_of_nodes(), len(self.abs_df_hashes)))

                nids_to_abs_df = self.abs_df[self.abs_df["graph_id"] == _id]
                nids_to_abs_df = nids_to_abs_df.set_index(nids_to_abs_df["node_id"].map(iddict))
                for nid in range(len(dgl_feat)):
                    f = nids_to_abs_df["hash"].get(nid, None)
                    dgl_feat[nid, self.abs_df_hashes.index(f)] = 1
                return dgl_feat
            g.ndata["_ABS_DATAFLOW"] = get_abs_dataflow_features(_id)


        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        g.edata["_ETYPE"] = th.Tensor(et).long()
        #emb_path = svd.cache_dir() / f"codebert_method_level/{_id}.pt"
        #g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        # print("compute")
        return g
        
    def cache_items(self, codebert):
        """Cache all items."""
        for i in tqdm(self.df.sample(len(self.df)).id.tolist(), desc="cache_items"):
            try:
                self.item(i, codebert)
            except Exception as E:
                print("cache_items exception", E)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert.

        ONLY NEEDS TO BE RUN ONCE.
        """
        savedir = svd.get_dir(svd.cache_dir() / "codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = svd.chunks((range(len(self.df))), 128)
        for idx_batch in tqdm(batches, desc="cache_codebert_method_level"):
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

    def __len__(self):
        """Get length of dataset."""
        return len(self.idx2id)

    def __iter__(self) -> dict:
        for i in self.idx2id:
            yield self[i]


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
        feat: str = "all",
        load_code=False,
        cache_all=False,
        undersample=True,
        filter_cwe=[],
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        dataargs = {
            "sample": sample, "gtype": gtype,
            "splits": splits, "feat": feat,
            "load_code": load_code, "cache_all": cache_all,
            "undersample": undersample, "filter_cwe": filter_cwe,
        }
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
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
            num_workers=4,
        )

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val))))
            return self.node_dl(g)
        return GraphDataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, batch_size=32, num_workers=0)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, batch_size=32, num_workers=0)
