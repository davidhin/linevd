import dgl
from dgl.dataloading import GraphDataLoader
import pytorch_lightning as pl

from sastvd.linevd.dataset import BigVulDatasetLineVD


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
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        dataargs = {
            "sample": -1,
            "gtype": gtype,
            # "splits": splits,
            "feat": feat,
            "load_code": load_code,
            "cache_all": cache_all,
            "undersample": resampling == "undersample",
            "filter_cwe": filter_cwe,
            "sample_mode": sample_mode,
            "use_cache": use_cache,
            "split": split,
        }
        self.feat = feat
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        duped_examples_trainval = set(self.train.df["id"]) & set(self.val.df["id"])
        assert not any(duped_examples_trainval), len(duped_examples_trainval)
        duped_examples_valtest = set(self.val.df["id"]) & set(self.test.df["id"])
        assert not any(duped_examples_valtest), len(duped_examples_valtest)
        print("SPLIT SIZES:", len(self.train),len(self.val),len(self.test))
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops
        self.train_workers = train_workers

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

    @staticmethod
    def add_model_specific_args(parent_parser):
        # parser.add_argument("--dataset", choices=all_datasets, required=True, help="short ID for the dataset to train on")
        parser.add_argument("--feat", help="node features to use")
        parser.add_argument("--gtype", help="node features to use")
        parser.add_argument("--batch_size", type=int, default=256, help="number of items to load in a batch")
        parser.add_argument("--filter", type=str, help="filter data to a certain persuasion", default="")
        parser.add_argument("--label_style", type=str, help="use node or graph labels", default="graph")
        parser.add_argument("--debug_train_batches", type=int, help="debug mode - train with n batches")
        parser.add_argument("--undersample_factor", type=float, help="factor to undersample majority class")
        parser.add_argument("--cache_all", action="store_true", help="cache all items in memory")
        parser.add_argument("--disable_cache", action="store_true", help="use cached files for dataset")
        parser.add_argument("--sample_mode", action="store_true", help="load only sample of dataset")
        parser.add_argument("--train_workers", type=int, default=4, help="use n parallel dataloader workers")
        parser.add_argument("--split", choices=["fixed", "random"], default="fixed", help="which split method to use")
        parser.add_argument("--filter_cwe", nargs="+", help="CWE to filter examples")
        parser.add_argument("--resampling", choices=["default", "undersample"], default="default", action="store_true", help="resampling mode")
        parser = parent_parser.add_argument_group("LineVD arguments (deprecated)")
        parser.add_argument("--nsampling", action="store_true")
        parser.add_argument("--nsampling_hops", type=int, default=1)

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
