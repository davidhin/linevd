import json
import os

import dgl
from dgl.data.utils import load_graphs, save_graphs
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.ivdetect.evaluate as ivde
import torch as th

from sastvd.linevd.utils import feature_extraction


class BigVulDatasetLineVD(svddc.BigVulDataset):
    """IVDetect version of BigVul."""

    def __init__(self, gtype="cfg", feat="all", cache_all=False, use_cache=True, catch_storage_errors=10, **kwargs):
        """Init."""
        self.graph_type = gtype
        self.feat = feat
        super(BigVulDatasetLineVD, self).__init__(feat=feat, **kwargs)
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
                # print("return from cache")
                return self.cache_all_cache[_id]
            else:
                # print("load into cache")
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

            def get_dataflow_1g_features(_id):
                dgl_feat = th.zeros((g.number_of_nodes(), self.df_1g_max_idx * 2))
                # dfg = self.df_1g.groupby("node_id")
                # dgl_feat = th.zeros((g.number_of_nodes(), dfg["gen"].count() + dfg["kill"].count()))
                
                nids_to_1g_df = self.df_1g_group.get_group(_id)
                node_id_dgl = nids_to_1g_df["node_id"].map(iddict)
                # all_nids = [i for i in node_id_dgl.dropna().astype(int).tolist() if i in iddict.values()]
                all_nids = (
                    node_id_dgl.dropna().astype(int).sort_values().unique().tolist()
                )
                nids_gen = dict(zip(node_id_dgl, nids_to_1g_df["gen"]))
                nids_kill = dict(zip(node_id_dgl, nids_to_1g_df["kill"]))
                try:
                    for nid in range(len(dgl_feat)):
                        for i in sorted(json.loads(nids_gen.get(nid, "[]"))):
                            if i in iddict and iddict[i] in all_nids:
                                dgl_feat[nid, all_nids.index(iddict[i])] = 1
                        for i in sorted(json.loads(nids_kill.get(nid, "[]"))):
                            if i in iddict and iddict[i] in all_nids:
                                dgl_feat[
                                    nid, self.df_1g_max_idx + all_nids.index(iddict[i])
                                ] = 1
                except Exception:
                    print(_id, nids_gen, nids_kill)
                    raise
                return dgl_feat

            g.ndata["_1G_DATAFLOW"] = get_dataflow_1g_features(_id)

        if "_ABS_DATAFLOW" in self.feat:
            if "all" in self.feat:
                hashes = self.abs_df_hashes["all"]
                # print(json.dumps(hashes, indent=2))
                dgl_feat = th.zeros((g.number_of_nodes(), len(hashes)))
                nids_to_abs_df = self.abs_df[self.abs_df["graph_id"] == _id]
                nids_to_abs_df = nids_to_abs_df.set_index(
                    nids_to_abs_df["node_id"].map(iddict)
                )
                for nid in range(len(dgl_feat)):
                    _hash = nids_to_abs_df["hash.all"].get(nid, None)
                    if _hash is not None:
                        # _hash = json.dumps(_hash)
                        idx = hashes.get(_hash, hashes[None])
                        # print(repr(_hash), idx, repr("{\"datatype\": [\"int\"]}"), hashes.get("{\"datatype\": [\"int\"]}"))
                        # print(repr(_hash), idx)#, repr("{\"datatype\": [\"int\"]}"), hashes.get("{\"datatype\": [\"int\"]}"))
                        dgl_feat[nid, idx] = 1
                g.ndata["_ABS_DATAFLOW"] = dgl_feat
            else:
                single = {
                    "api": False,
                    "datatype": True,
                    "literal": False,
                    "operator": False,
                }

                def get_abs_dataflow_features(_id):
                    dgl_feats = []
                    for subkey in ["api", "datatype", "literal", "operator"]:
                        if subkey not in self.feat:
                            continue
                        hashes = self.abs_df_hashes[subkey]
                        dgl_feat = th.zeros((g.number_of_nodes(), len(hashes)))
                        hash_name = f"hash.{subkey}"

                        nids_to_abs_df = self.abs_df[self.abs_df["graph_id"] == _id]
                        nids_to_abs_df = nids_to_abs_df.set_index(
                            nids_to_abs_df["node_id"].map(iddict)
                        )
                        for nid in range(len(dgl_feat)):
                            # Flip the bit for a single value
                            if single[subkey]:
                                f = nids_to_abs_df[hash_name].get(nid, None)
                                if f is not None:
                                    idx = hashes[f] if f in hashes else hashes[None]
                                    dgl_feat[nid, idx] = 1
                            # Flip the bit for all values present
                            else:
                                for f in nids_to_abs_df[hash_name].get(nid, []):
                                    idx = hashes.get(f, hashes[None])
                                    dgl_feat[nid, idx] = 1
                        dgl_feats.append(dgl_feat)
                    return th.cat(dgl_feats, axis=1)
                g.ndata["_ABS_DATAFLOW"] = get_abs_dataflow_features(_id)

            def get_abs_dataflow_kill_features(_id):
                # print(nids_to_1g_df)
                gen = g.ndata["_ABS_DATAFLOW"]
                dgl_feats = th.zeros((g.number_of_nodes(), gen.shape[1]))
                nids_to_1g_df = self.df_1g_group.get_group(_id)
                nids_to_1g_df_dgl = nids_to_1g_df[["node_id", "kill"]].copy()
                nids_to_1g_df_dgl = nids_to_1g_df_dgl.assign(
                    node_id=nids_to_1g_df_dgl["node_id"].map(iddict),
                    kill=nids_to_1g_df_dgl["kill"].apply(lambda k: [iddict[ki] for ki in k if ki in iddict]),
                ).dropna()
                nids_to_1g_df_dgl["node_id"] = nids_to_1g_df_dgl["node_id"].apply(int)
                nids_kill = nids_to_1g_df_dgl.set_index("node_id")["kill"].to_dict()
                for nid in range(len(dgl_feats)):
                    if nid in nids_kill:
                        for kill_nid in nids_kill:
                            dgl_feats[nid] = dgl_feats[nid] + gen[kill_nid]
                return dgl_feats
            if "abskill" in self.feat:
                g.ndata["_ABS_DATAFLOW_kill"] = get_abs_dataflow_kill_features(_id)
            if "edgekill" in self.feat:
                nids_to_1g_df = self.df_1g_group.get_group(_id)
                nids_to_1g_df_dgl = nids_to_1g_df[["node_id", "kill"]].copy()
                nids_to_1g_df_dgl = nids_to_1g_df_dgl.explode("kill")
                nids_to_1g_df_dgl = nids_to_1g_df_dgl.assign(
                    node_id=nids_to_1g_df_dgl["node_id"].map(iddict),
                    kill=nids_to_1g_df_dgl["kill"].map(iddict),
                ).dropna()
                nids_to_1g_df_dgl["node_id"] = nids_to_1g_df_dgl["node_id"].apply(int)
                nids_to_1g_df_dgl["kill"] = nids_to_1g_df_dgl["kill"].apply(int)
                # print(nids_to_1g_df_dgl)
                et.extend([max(et) + 1] * len(nids_to_1g_df_dgl))
                # print(nids_to_1g_df_dgl.dtypes)
                # print(et)
                g.add_edges(nids_to_1g_df_dgl["node_id"].tolist(), nids_to_1g_df_dgl["kill"].tolist())

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
    for i, d in enumerate(ds):
        if i >= 10:
            break
        print(i, d)
        print(d.number_of_nodes(), d.ndata["_ABS_DATAFLOW"].sum().item(), d.ndata["_ABS_DATAFLOW"][:, 1:].sum().item())

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
