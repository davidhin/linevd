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

    def __init__(self, partition="train", vulonly=False, sample=-1, splits="default", load_code=False):
        """Init class."""
        # Get finished samples
        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(svd.processed_dir() / "bigvul/before/*nodes*"))
        ]
        self.df = svdds.bigvul(splits=splits)
        self.partition = partition
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]
        # NOTE: drop several columns to save memory
        if not load_code:
            self.df = self.df.drop(columns=["before", "after", "removed", "added", "diff"])
        print("len(df)=", len(self.df))
        print("df head=", self.df.head())

        # Balance training set
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Correct ratio for test set
        if partition == "test":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul.sample(min(len(nonvul), len(vul) * 20), random_state=0)
            self.df = pd.concat([vul, nonvul])

        # Small sample (for debugging):
        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)

        # Filter only vulnerable
        if vulonly:
            self.df = self.df[self.df.vul == 1]

        # Filter out samples with no lineNumber from Joern output
        valid_cache = svd.cache_dir() / f"bigvul_valid_{partition}.csv"
        if valid_cache.exists():
            valid_cache_df = pd.read_csv(valid_cache, index_col=0)
            # print(valid_cache_df.head(), self.df.head())
        else:
            valid = svd.dfmp(
                self.df, BigVulDataset.check_validity, "id", desc="Validate Samples: "
            )
            df_id = self.df.id
            valid_cache_df = pd.DataFrame({"id": df_id, "valid": valid}, index=self.df.index)
            valid_cache_df.to_csv(valid_cache)
        print(valid_cache_df["valid"].value_counts())
        self.df = self.df[self.df.id.isin(valid_cache_df[valid_cache_df["valid"]].id)]

        # Load abstract dataflow information
        abs_df_file = svd.processed_dir() / f"bigvul/abstract_dataflow_hash_sample=False.csv"
        if abs_df_file.exists():
            self.abs_df = pd.read_csv(abs_df_file)
        else:
            print("YOU SHOULD RUN abstract_dataflow.py")

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        # Load Glove vectors.
        # glove_path = svd.processed_dir() / "bigvul/glove_False/vectors.txt"
        # self.emb_dict, _ = svdglove.glove_dict(glove_path)

    def itempath(_id):
        """Get itempath path from item id."""
        return svd.processed_dir() / f"bigvul/before/{_id}.c"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"bigvul/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(BigVulDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(BigVulDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print("valid exception", traceback.format_exc(), str(BigVulDataset.itempath(_id)))
            return False

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
