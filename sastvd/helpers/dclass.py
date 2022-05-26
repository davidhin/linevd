import json
import traceback
from glob import glob
from pathlib import Path

import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdds
import sastvd.helpers.glove as svdglove


class BigVulDataset:
    """Represent BigVul as graph dataset."""

    def __init__(
        self, partition="train", splits="default",
        check_file=True, check_valid=True, vulonly=False, load_code=False, sample=-1
        ):
        """Init class."""
        # Get finished samples
        self.partition = partition
        
        df = svdds.bigvul(splits=splits)
        df = svdds.bigvul_filter(df, check_file=check_file, check_valid=check_valid, vulonly=vulonly, load_code=load_code, sample=sample)
        df = svdds.bigvul_partition(df, partition)

        self.df = df

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()
        
        self.abs_df, self.abs_df_hashes = svdds.abs_dataflow()
        self.df_1g = svdds.dataflow_1g()
        self.df_1g_max_idx = max(max(max(int(s) if s.isdigit() else -1 for s in l.split(",")) for l in self.df_1g[k]) for k in ["gen", "kill"])

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"BigVulDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"
