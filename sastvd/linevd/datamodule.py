import dgl
from dgl.dataloading import GraphDataLoader
import pytorch_lightning as pl

from sastvd.linevd.dataset import BigVulDatasetLineVD
import functools
import traceback

import sastvd as svd
import sastvd.helpers.datasets as svdds
import torch as th
import json

class DF1G:
    def __init__(self, sample_mode, verbose=False):
        try:
            df_1g = svdds.dataflow_1g(sample_mode)
            df_1g_group = df_1g.groupby("graph_id")
            nuniq_nodes = df_1g.groupby("graph_id")["node_id"].nunique()
            too_large_idx = nuniq_nodes[nuniq_nodes > 500].index

            df_1g = df_1g[~self.df_1g["graph_id"].isin(too_large_idx)]
            df_1g_max_idx = max(
                df_1g.groupby("graph_id")["node_id"].nunique()
            )
            if verbose:
                print("df_1g_max_idx =", df_1g_max_idx)

            self.df_1g = df_1g
            self.df_1g_group = df_1g_group
        except Exception:
            print("could not load 1G features")
            traceback.print_exc()
    
    def get_features(self, _id, g, iddict):
        dgl_feat = th.zeros((g.number_of_nodes(), self.df_1g_max_idx * 2))
        
        nids_to_1g_df = self.df_1g_group.get_group(_id)
        node_id_dgl = nids_to_1g_df["node_id"].map(iddict)
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

class DFABS:
    def __init__(self, feat, sample_mode, split, seed):
        self.feat = feat
        self.abs_df, self.abs_df_hashes = svdds.abs_dataflow(feat, sample_mode, split=split, seed=seed)
    
    def get_features(self, _id, g, iddict):
        dgl_feat = None
        if "all" in self.feat:
            hashes = self.abs_df_hashes["all"]
            dgl_feat = th.zeros((g.number_of_nodes(), len(hashes)))
            nids_to_abs_df = self.abs_df[self.abs_df["graph_id"] == _id]
            nids_to_abs_df = nids_to_abs_df.set_index(
                nids_to_abs_df["node_id"].map(iddict)
            )
            for nid in range(len(dgl_feat)):
                _hash = nids_to_abs_df["hash.all"].get(nid, None)
                if _hash is not None:
                    idx = hashes.get(_hash, hashes[None])
                    dgl_feat[nid, idx] = 1
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
            dgl_feat = get_abs_dataflow_features(_id)

        # def get_abs_dataflow_kill_features(_id, dgl_feat):
        #     gen = dgl_feat
        #     dgl_feats = th.zeros((g.number_of_nodes(), gen.shape[1]))
        #     nids_to_1g_df = self.df_1g_group.get_group(_id)
        #     nids_to_1g_df_dgl = nids_to_1g_df[["node_id", "kill"]].copy()
        #     nids_to_1g_df_dgl = nids_to_1g_df_dgl.assign(
        #         node_id=nids_to_1g_df_dgl["node_id"].map(iddict),
        #         kill=nids_to_1g_df_dgl["kill"].apply(lambda k: [iddict[ki] for ki in k if ki in iddict]),
        #     ).dropna()
        #     nids_to_1g_df_dgl["node_id"] = nids_to_1g_df_dgl["node_id"].apply(int)
        #     nids_kill = nids_to_1g_df_dgl.set_index("node_id")["kill"].to_dict()
        #     for nid in range(len(dgl_feats)):
        #         if nid in nids_kill:
        #             for kill_nid in nids_kill:
        #                 dgl_feats[nid] = dgl_feats[nid] + gen[kill_nid]
        #     return dgl_feats
        # if "abskill" in self.feat:
        #     dgl_feat = get_abs_dataflow_kill_features(_id, dgl_feat)
        # if "edgekill" in self.feat:
        #     nids_to_1g_df = self.df_1g_group.get_group(_id)
        #     nids_to_1g_df_dgl = nids_to_1g_df[["node_id", "kill"]].copy()
        #     nids_to_1g_df_dgl = nids_to_1g_df_dgl.explode("kill")
        #     nids_to_1g_df_dgl = nids_to_1g_df_dgl.assign(
        #         node_id=nids_to_1g_df_dgl["node_id"].map(iddict),
        #         kill=nids_to_1g_df_dgl["kill"].map(iddict),
        #     ).dropna()
        #     nids_to_1g_df_dgl["node_id"] = nids_to_1g_df_dgl["node_id"].apply(int)
        #     nids_to_1g_df_dgl["kill"] = nids_to_1g_df_dgl["kill"].apply(int)
        #     et.extend([max(et) + 1] * len(nids_to_1g_df_dgl))
        #     g.add_edges(nids_to_1g_df_dgl["node_id"].tolist(), nids_to_1g_df_dgl["kill"].tolist())
        return dgl_feat

class BigVulDatasetLineVDDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Bigvul."""

    def __init__(
        self,
        feat,
        gtype,
        load_code=True,
        resampling="none",
        filter_cwe="",
        sample_mode=False,
        cache_all=True,
        use_cache=True,
        train_workers=4,
        split="fixed",
        batch_size=256,
        nsampling=False,
        nsampling_hops=1,
        seed=0,
        verbose=False,
        test_every=False,
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        dataargs = {
            "sample": -1,
            "gtype": gtype,
            "feat": feat,
            "load_code": load_code,
            "cache_all": cache_all,
            "undersample": resampling == "undersample",
            "filter_cwe": filter_cwe,
            "sample_mode": sample_mode,
            "use_cache": use_cache,
            "split": split,
            "seed": seed,
            "verbose": verbose,
        }
        self.feat = feat
        embedders = self.get_embedders(feat, sample_mode, split, seed, verbose)
        self.train = BigVulDatasetLineVD(partition="train", embedders=embedders, **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", embedders=embedders, **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", embedders=embedders, **dataargs)
        duped_examples_trainval = set(self.train.df["id"]) & set(self.val.df["id"])
        assert not any(duped_examples_trainval), len(duped_examples_trainval)
        duped_examples_valtest = set(self.val.df["id"]) & set(self.test.df["id"])
        assert not any(duped_examples_valtest), len(duped_examples_valtest)
        if verbose:
            print("SPLIT SIZES:", len(self.train),len(self.val),len(self.test))
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops
        self.train_workers = train_workers
        self.test_every = test_every
    
    def get_embedders(self, feat, sample_mode, split, seed, verbose):
        embedders = {}
        if "_ABS_DATAFLOW" in feat:
            embedders["_ABS_DATAFLOW"] = DFABS(feat, sample_mode, split, seed)
        if "_1G_DATAFLOW" in feat:
            embedders["_1G_DATAFLOW"] = DF1G(feat, sample_mode, split, seed, verbose)
        return embedders

    @property
    def input_dim(self):
        if self.feat.startswith("_ABS_DATAFLOW"):
            featname = "_ABS_DATAFLOW"
        else:
            featname = self.feat
        return self.train[0].ndata[featname].shape[1]

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
        return GraphDataLoader(
            self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.train_workers,
        )

    def val_dataloader(self):
        if self.test_every:
            return [self._val_dataloader(), self.test_dataloader()]
        else:
            return self._val_dataloader()

    def _val_dataloader(self):
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
        return GraphDataLoader(self.test, batch_size=32, num_workers=self.train_workers)

def test_dm():
    data = BigVulDatasetLineVDDataModule(
        batch_size=256,
        methodlevel=False,
        gtype="cfg",
        feat="_ABS_DATAFLOW_datatype_all",
        cache_all=False,
        undersample=True,
        filter_cwe=[],
        sample_mode=False,
        use_cache=True,
        train_workers=0,
        split="random",
    )
    print(data)
