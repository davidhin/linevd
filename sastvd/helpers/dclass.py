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
        self,
        partition="train",
        embedders=None,
        check_file=True,
        check_valid=True,
        vulonly=False,
        load_code=False,
        sample=-1,
        undersample=True,
        filter_cwe=None,
        sample_mode=False,
        split="fixed",
        seed=0,
        verbose=False,
    ):
        """Init class."""
        # Get finished samples
        self.partition = partition

        df = svdds.bigvul(sample=sample_mode, verbose=verbose)
        if sample != -1:
            df = df.sample(sample, random_state=seed)
        # print("load", len(df))
        # Filter to storage/cache/bigvul_linevd_codebert_dataflow_cfg
        # df = df[df["id"].apply(lambda i: (Path("storage/cache/bigvul_linevd_codebert_dataflow_cfg")/str(i)).exists())]
        # print("original", len(df))
        df = svdds.bigvul_filter(
            df,
            check_file=check_file,
            check_valid=check_valid,
            vulonly=vulonly,
            load_code=load_code,
            sample=sample,
            sample_mode=sample_mode,
            verbose=verbose,
        )
        # print("svdds.bigvul_filter", len(df))

        if filter_cwe:
            md = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
            mdf = pd.merge(df, md, on="id")
            npd_idx = mdf[mdf["CWE ID"].isin(filter_cwe)]["id"]
            df = df[df["id"].isin(npd_idx)]
            if verbose:
                print("CWE filter", len(df))

        if not sample_mode:
            df = svdds.bigvul_partition(df, partition, undersample=undersample, split=split, seed=seed, verbose=verbose)
        if verbose:
            print(partition, len(df))

        self.df = df

        # Get mapping from index to sample ID.
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

        self.embedders = embedders

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


def test_1g():
    BigVulDataset(feat="_1G_DATAFLOW")

def test_abs():
    ds = BigVulDataset(feat="_ABS_DATAFLOW_api_datatype_literal_operator", sample_mode=True)
    print(ds)
    for i, d in enumerate(ds):
        print(i, d)


def test_abs_ds():
    BigVulDataset(feat="_ABS_DATAFLOW_filtertoabs", sample=1000)


def test_abs_npd():
    BigVulDataset(feat="_ABS_DATAFLOW", filter_cwe=["CWE-476", "CWE-690"])
