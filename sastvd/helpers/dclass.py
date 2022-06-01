import functools
import traceback
import tqdm

import pandas as pd
import sastvd as svd
import sastvd.helpers.datasets as svdds
import sastvd.helpers.joern as svdj

def is_valid(_id, hash_index):
    n, e = svdj.get_node_edges(svdds.itempath(_id))
    e = svdj.rdg(e, "cfg")
    n = svdj.drop_lone_nodes(n, e)
    n_hashed = n["id"].apply(lambda nid: hash_index.get((_id, nid), -1) != -1)
    return n_hashed.sum() > 0

class BigVulDataset:
    """Represent BigVul as graph dataset."""

    def __init__(
        self, partition="train", splits="default",
        check_file=True, check_valid=True, vulonly=False, load_code=False, sample=-1, undersample=True,
        feat="all", filter_cwe=None,
        ):
        """Init class."""
        # Get finished samples
        self.partition = partition
        
        df = svdds.bigvul(splits=splits)
        if sample != -1:
            df = df.sample(sample, random_state=0)
        # print("load", len(df))
        # Filter to storage/cache/bigvul_linevd_codebert_dataflow_cfg
        # df = df[df["id"].apply(lambda i: (Path("storage/cache/bigvul_linevd_codebert_dataflow_cfg")/str(i)).exists())]
        # print("original", len(df))
        df = svdds.bigvul_filter(df, check_file=check_file, check_valid=check_valid, vulonly=vulonly, load_code=load_code, sample=sample)
        print("svdds.bigvul_filter", len(df))
        
        if "_ABS_DATAFLOW" in feat:
            try:
                self.abs_df, self.abs_df_hashes = svdds.abs_dataflow()
                if "_filtertoabs" in feat:
                    filtered_file = svd.processed_dir() / f"bigvul/abstract_dataflow_hash_all_filtertoabs.csv"
                    if filtered_file.exists():
                        valid_df = pd.read_csv(filtered_file, index_col="graph_id")
                    else:
                        hash_index = self.abs_df.set_index(["graph_id", "node_id"])["hash"]
                        df["valid"] = svd.dfmp(df, functools.partial(is_valid, hash_index=hash_index), columns="id", workers=6, desc="filter abs df to known")
                        valid_df = df[["id", "valid"]].rename(columns={"id": "graph_id"}).set_index("graph_id")
                        valid_df.to_csv(filtered_file)
                    print(valid_df)
                    print("valid check", valid_df["valid"].value_counts())
                    df = df[df["id"].map(valid_df["valid"].to_dict())]
                    print("ABS filter", len(df))
            except Exception:
                print("could not load abstract features")
                traceback.print_exc()

        if "_1G_DATAFLOW" in feat:
            try:
                self.df_1g = svdds.dataflow_1g()
                # self.df_1g_max_idx = max(max(max(int(s) if s.isdigit() else -1 for s in l.split(",")) for l in self.df_1g[k]) for k in ["gen", "kill"])
                # breakpoint()
                nuniq_nodes = self.df_1g.groupby("graph_id")["node_id"].nunique()
                # percentile_99 = nuniq_nodes.quantile([.99]).item()
                # too_large_idx = nuniq_nodes[nuniq_nodes > percentile_99].index
                too_large_idx = nuniq_nodes[nuniq_nodes > 500].index

                self.df_1g = self.df_1g[~self.df_1g["graph_id"].isin(too_large_idx)]
                self.df_1g_max_idx = max(self.df_1g.groupby("graph_id")["node_id"].nunique())
                print("self.df_1g_max_idx =", self.df_1g_max_idx)
                
                df = df[df["id"].isin(set(self.df_1g["graph_id"]))]
            except Exception:
                print("could not load 1G features")
                traceback.print_exc()
            print("1G filter", len(df))

        if filter_cwe:
            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            mdf = pd.merge(df, md, on="id")
            npd_idx = mdf[mdf["CWE ID"].isin(filter_cwe)]["id"]
            df = df[df["id"].isin(npd_idx)]
            print("CWE filter", len(df))

        df = svdds.bigvul_partition(df, partition, undersample=undersample)
        print(partition, len(df))

        self.df = df

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

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

def test_ds():
    BigVulDataset(feat="_1G_DATAFLOW")
def test_abs_ds():
    BigVulDataset(feat="_ABS_DATAFLOW_filtertoabs", sample=1000)
def test_abs_npd():
    BigVulDataset(feat="_ABS_DATAFLOW", filter_cwe=["CWE-476", "CWE-690"])
