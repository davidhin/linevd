import json
import os

import dgl
from dgl.data.utils import load_graphs, save_graphs
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.ivdetect.evaluate as ivde
import torch as th
import tqdm
import time

from sastvd.linevd.utils import feature_extraction


class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="cfg", feat="all", cache_all=False, use_cache=True, catch_storage_errors=10, **kwargs):
        """Init."""
        self.graph_type = gtype
        self.feat = feat
        super(BigVulDatasetLineVD, self).__init__(**kwargs)
        lines = ivde.get_dep_add_lines_bigvul()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.cache_all = cache_all
        self.cache_all_cache = {}
        self.lines = lines
        self.use_cache = use_cache
        self.catch_storage_errors = catch_storage_errors

    def item(self, _id, must_load=False, use_cache=True):
        """Cache item."""

        if self.cache_all and not must_load and use_cache:
            if _id in self.cache_all_cache:
                return self.cache_all_cache[_id]
            else:
                g = self.item(_id, must_load=True, use_cache=use_cache)
                self.cache_all_cache[_id] = g
                return g

        savedir = svd.get_dir(
            svd.cache_dir() / f"bigvul_linevd_codebert_{self.graph_type}_{self.feat}"
        ) / str(_id)
        if os.path.exists(savedir) and use_cache:
            try:
                g = load_graphs(str(savedir))[0][0]
            except Exception:
                savedir.unlink()
                return self.item(_id, must_load=must_load)
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
            if "_1G_DATAFLOW" in g.ndata:
                if g.ndata["_1G_DATAFLOW"].size(1) != self.df_1g_max_idx * 2:
                    savedir.unlink()
                    return self.item(_id, must_load=True)

            return g
        code, lineno, ei, eo, et, nids, ntypes, iddict = feature_extraction(
            svddc.svdds.itempath(_id),
            self.graph_type,
            return_node_ids=True,
            return_iddict=True,
            group=False,
            return_node_types=True,
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
            g.ndata["_1G_DATAFLOW"] = self.embedders["_1G_DATAFLOW"].get_features(_id, g, iddict)

        if "_ABS_DATAFLOW" in self.feat:
            g.ndata["_ABS_DATAFLOW"] = self.embedders["_ABS_DATAFLOW"].get_features(_id, g, iddict)

        if "CODEBERT" in self.feat:
            code = [c.replace("\\t", "").replace("\\n", "") for c in code]
            chunked_batches = svd.chunks(code, 128)
            features = [codebert.encode(c).detach().cpu() for c in chunked_batches]
            g.ndata["_CODEBERT"] = th.cat(features)

        g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
        g.edata["_ETYPE"] = th.Tensor(et).long()
        g = dgl.add_self_loop(g)
        save_graphs(str(savedir), [g])
        return g

    def __getitem__(self, idx):
        """Override getitem."""
        tries = 0
        while True:
            try:
                tries += 1
                return self.item(self.idx2id[idx], use_cache=self.use_cache)
            except (BrokenPipeError, OSError):
                print("index", idx, "tried", tries, "times out of", self.catch_storage_errors)
                # Catch common storage errors
                if tries > self.catch_storage_errors:
                    raise
                time.sleep(3)

    def __len__(self):
        """Get length of dataset."""
        return len(self.idx2id)

    def __iter__(self) -> dict:
        for i in self.idx2id:
            yield self[i]

def test_abs():
    ds = BigVulDatasetLineVD(feat="_ABS_DATAFLOW_api_datatype_literal_operator", partition="all", sample_mode=False, use_cache=False)
    print(ds)
    for i, d in enumerate(ds):
        if i >= 10:
            break
        print(i, d)

def test_abs_datatype():
    ds = BigVulDatasetLineVD(feat="_ABS_DATAFLOW_datatype", partition="all", sample_mode=False, use_cache=False)
    print(ds)
    for i, d in enumerate(ds):
        if i >= 10:
            break
        print(i, d)

def test_abs_datatype_abskill():
    ds = BigVulDatasetLineVD(feat="_ABS_DATAFLOW_datatype_abskill", partition="all", sample_mode=False, use_cache=False)
    print(ds)
    for i, d in enumerate(ds):
        if i >= 10:
            break
        print(i, d)
        print(d.number_of_nodes(), d.ndata["_ABS_DATAFLOW"].sum().item(), d.ndata["_ABS_DATAFLOW"][:, 1:].sum().item())
        print(d.number_of_nodes(), d.ndata["_ABS_DATAFLOW_kill"].sum().item(), d.ndata["_ABS_DATAFLOW_kill"][:, 1:].sum().item())

def test_abs_datatype_edgekill():
    ds = BigVulDatasetLineVD(feat="_ABS_DATAFLOW_datatype_edgekill", partition="all", sample_mode=False, use_cache=False)
    print(ds)
    for i, d in enumerate(ds):
        if i >= 10:
            break
        print(i, d)

def test_abs_datatype_hash():
    ds = BigVulDatasetLineVD(feat="_ABS_DATAFLOW_datatype_all", partition="all", sample_mode=False, use_cache=False)
    print(ds)
    with open("tensor.txt", "w") as of:
        for i, d in enumerate(tqdm.tqdm(ds, desc="get dataset examples")):
            if i < 10:
                print(i, d)
                print(d.number_of_nodes(), d.ndata["_ABS_DATAFLOW"].sum().item(), d.ndata["_ABS_DATAFLOW"][:, 1:].sum().item())
                th.set_printoptions(profile="full")
                print(d.ndata["_ABS_DATAFLOW"].argmax(dim=1), file=of)
            row_sums = d.ndata["_ABS_DATAFLOW"].sum(dim=1)
            assert th.all(th.eq(row_sums, th.zeros_like(row_sums)) | th.eq(row_sums, th.ones_like(row_sums)))

def test_abs_all_hash():
    ds = BigVulDatasetLineVD(feat="_ABS_DATAFLOW_api_datatype_literal_operator_all", partition="all", sample_mode=False, use_cache=False)
    print(ds)
    for i, d in enumerate(ds):
        if i >= 10:
            break
        print(i, d)
        print(d.number_of_nodes(), d.ndata["_ABS_DATAFLOW"].sum().item(), d.ndata["_ABS_DATAFLOW"][:, 1:].sum().item())

def test_1g():
    ds = BigVulDatasetLineVD(feat="_1G_DATAFLOW", partition="all", sample_mode=True)
    print(ds)
    for i, d in enumerate(ds):
        print(i, d)
