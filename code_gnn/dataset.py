import logging
import os
import pickle

import numpy as np
import pandas as pd
from dgl.data import DGLDataset, split_dataset
from torch.utils.data import ConcatDataset, Subset

from code_gnn.cpg_cache import CPGCache
from code_gnn.dataloader import MyDataLoader
from code_gnn.globals import ml_data_dir, get_cache_filename
import re

logger = logging.getLogger(__name__)


class MyDGLDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to show progress information
    """

    def __init__(self,
                 config,
                 name=None,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.model_type = config["model"]
        self.node_limit = config["node_limit"]
        self.graph_limit = config["graph_limit"]
        self.label_style = config["label_style"]

        # self.graphs = None
        # self.labels = None

        if raw_dir is None:
            raw_dir = ml_data_dir
        if name is None:
            name = config["dataset"]

        # New
        data_dir = raw_dir / name
        self.dgl_cache = CPGCache(data_dir / 'cache_dgl')

        df = pd.read_pickle(data_dir / (name + '_with_dgl.pkl'))
        logger.debug(f'Original len(df)={len(df)}')
        # df = [df["dgl_graph_outcome"] == 'success']
        df = df[[self.dgl_cache.get_cpg_filepath(i).exists() for i in df.index]]
        logger.info(f'After remove invalid len(df)={len(df)}')
        if config["filter"] is not None:
            if "small" in config["filter"]:
                try:
                    df = pd.concat((df[df["target"] == 1].head(50), df[df["target"] == 0].head(50)))
                except KeyError:
                    df = df.head(100)
                logger.info(f'Filter sample len(df)={len(df)}')
            if "npd" in config["filter"]:
                if name == "SARD":
                    df["label_names"] = df["labels"].apply(lambda labels: {l["name"] for l in labels})
                    df["label_names_str"] = df["label_names"].apply(str)
                    df["CWE ID"] = df["label_names_str"].apply(lambda s: ",".join(re.findall(r'(CWE-[0-9]+)', s)))
                    df = df[df["CWE ID"].str.contains("CWE-476") | df["CWE ID"].str.contains("CWE-690")]  # NPD only
                elif name == "MSR":
                    df = df[df["CWE ID"].isin(["CWE-476", "CWE-690"])]  # NPD only
                logger.info(f'Filter npd len(df)={len(df)}')
            if "bof" in config["filter"]:
                if name == "SARD":
                    df["label_names"] = df["labels"].apply(lambda labels: {l["name"] for l in labels})
                    df["label_names_str"] = df["label_names"].apply(str)
                    df["CWE ID"] = df["label_names_str"].apply(lambda s: ",".join(re.findall(r'(CWE-[0-9]+)', s)))
                    df = df[df["CWE ID"].str.contains("CWE-121") | df["CWE ID"].str.contains("CWE-122") | df["CWE ID"].str.contains("CWE-124")]  # NPD only
                    """
                    CWE-121
                    CWE-122
                    CWE-124
                    """
            if "undersample_unrealistic" in config["filter"]:
                vdf = df[df["target"] == 1]
                ndf = df[df["target"] == 0]
                df = pd.concat((vdf, ndf.sample(len(vdf))))
        try:
            logger.info(df["target"])
            logger.info(df["target"].value_counts())
        except KeyError:
            pass
        self.df = df
        self.idx_to_row_name = {i: index for i, index in enumerate(self.df.index)}
        # self.shuffle()

        super(MyDGLDataset, self).__init__(name=name,
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def shuffle(self):
        idx_to_row_name_tuples = list(self.idx_to_row_name.items())
        indices = np.arange(0, len(idx_to_row_name_tuples))
        np.random.shuffle(indices)
        self.idx_to_row_name = dict([idx_to_row_name_tuples[i] for i in indices])

    def __getitem__(self, idx):
        row_name = self.idx_to_row_name[idx]
        g = self.dgl_cache.load_cpg(row_name)
        # Only for debug
        if self.name == "MSR":
            node_label = g.ndata['graph_label']
            graph_label = g.ndata['node_label']
            if len(graph_label.shape) > 1:
                # graph label was accidentally squared
                graph_label = graph_label[:, 0]
                # write fixed graph
                g.ndata['node_label'] = node_label
                g.ndata['graph_label'] = graph_label
                # logger.debug(f'borked and fixed {idx=} {g=}')
        elif self.name == "SARD":
            if len(g.ndata['graph_label'].shape) > 1:
                g.ndata['graph_label'] = g.ndata['graph_label'][:, 0]
        # Only for debug
        return g

    def __len__(self):
        # number of data examples
        return len(self.idx_to_row_name)

    @property
    def input_dim(self):
        return self[0].ndata['h'].shape[-1]

    @property
    def vulnerable(self):
        return Subset(
            self,
            np.array([i for i in range(len(self)) if self.df.loc[self.idx_to_row_name[i]]["target"] == 1])
        )

    @property
    def nonvulnerable(self):
        return Subset(
            self,
            np.array([i for i in range(len(self)) if self.df.loc[self.idx_to_row_name[i]]["target"] == 0])
        )

    def roll(self, n):
        """Roll dataset right by n items."""
        idx_to_row_name_tuples = list(self.idx_to_row_name.items())
        indices = np.arange(0, len(idx_to_row_name_tuples))
        np.roll(indices, n)
        self.idx_to_row_name = dict([idx_to_row_name_tuples[i] for i in indices])

    def process(self):
        pass

    def __repr__(self):
        return str(self)

    def __str__(self):
        inner_text = ','.join(
            [f'len(self)={len(self)} {self.input_dim}'])
        return f'{type(self).__name__}({inner_text})'


def split(filter, batch_size, dataset, dataset_splits):
    if "stratified" in filter:
        train_set_v, valid_set_v, test_set_v = split_dataset(dataset.vulnerable, frac_list=dataset_splits, shuffle=True)
        train_set_nv, valid_set_nv, test_set_nv = split_dataset(dataset.nonvulnerable, frac_list=dataset_splits, shuffle=True)
        if "undersample_realistic" in filter:
            indices = np.arange(len(train_set_nv))
            indices = indices[:len(train_set_v)]
            train_set_nv = Subset(train_set_nv, indices)
        train_set = ConcatDataset((train_set_v, train_set_nv))
        valid_set = ConcatDataset((valid_set_v, valid_set_nv))
        test_set = ConcatDataset((test_set_v, test_set_nv))
    else:
        train_set, valid_set, test_set = split_dataset(dataset, frac_list=dataset_splits, shuffle=True)
    train_dataloader = MyDataLoader(
        dataset=train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
    )
    valid_dataloader = MyDataLoader(
        dataset=valid_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
    )
    test_dataloader = MyDataLoader(
        dataset=test_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
    )
    return train_dataloader, valid_dataloader, test_dataloader
