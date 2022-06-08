import dgl
from dgl.dataloading import GraphDataLoader
import pytorch_lightning as pl

from sastvd.linevd.dataset import BigVulDatasetLineVD


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
        sample_mode=False,
        use_cache=True,
    ):
        """Init class from bigvul dataset."""
        super().__init__()
        dataargs = {
            "sample": sample,
            "gtype": gtype,
            # "splits": splits,
            "feat": feat,
            "load_code": load_code,
            "cache_all": cache_all,
            "undersample": undersample,
            "filter_cwe": filter_cwe,
            "sample_mode": sample_mode,
            "use_cache": use_cache,
        }
        self.train = BigVulDatasetLineVD(partition="train", **dataargs)
        self.val = BigVulDatasetLineVD(partition="val", **dataargs)
        self.test = BigVulDatasetLineVD(partition="test", **dataargs)
        print("SPLIT SIZES:", len(self.train),len(self.val),len(self.test))
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
        return GraphDataLoader(
            self.train, shuffle=True, batch_size=self.batch_size, num_workers=4
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
        return GraphDataLoader(self.test, batch_size=32, num_workers=0)
