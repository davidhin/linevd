from collections import defaultdict

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.linevd as lvd
from tqdm import tqdm


class EmpEvalBigVul:
    """Perform Empirical Evaluation."""

    def __init__(self, all_funcs: list, test_data: lvd.BigVulDatasetLineVD):
        """Init.

        Args:
            all_funcs (list): All funcs predictions returned from LitGNN.
            test_data (lvd.BigVulDatasetLineVD): Test set.

        Example:
            model = lvd.LitGNN()
            model = lvd.LitGNN.load_from_checkpoint($BESTMODEL$, strict=False)
            trainer.test(model, data)
            all_funcs = model.all_funcs

            datamodule_args = {"batch_size": 1024, "nsampling_hops": 2, "gtype": "pdg+raw"}

            eebv = EmpEvalBigVul(model.all_funcs, data.test)
            eebv.eval_test()
        """
        self.func_df = pd.read_csv(svd.cache_dir() / "bigvul/bigvul_metadata.csv")
        self.func_df = self.func_df.set_index("id")
        self.all_funcs = all_funcs
        self.test_data = test_data

    def func_metadata(self, _id):
        """Get func metadata."""
        return self.func_df.loc[_id].to_dict()

    def stmt_metadata(self, _id):
        """Get statement metadata."""
        n = lvd.feature_extraction(svddc.BigVulDataset.itempath(_id), return_nodes=True)
        keepcols = ["_label", "name", "controlStructureType", "local_type"]
        n = n.set_index("lineNumber")[keepcols]
        return n.to_dict("index")

    def test_item(self, idx):
        """Get test item information."""
        _id = self.test_data.idx2id[idx]
        preds = self.all_funcs[idx]
        f_data = self.func_metadata(_id)
        s_data = self.stmt_metadata(_id)

        # Format func data
        f_data["pred"] = preds[2].max().item()
        f_data["vul"] = max(preds[1])

        # Format statement data
        s_pred_data = defaultdict(dict)

        for i in range(len(preds[0])):
            s_pred_data[preds[3][i]]["vul"] = preds[1][i]
            s_pred_data[preds[3][i]]["pred"] = list(preds[0][i])
            s_pred_data[preds[3][i]].update(s_data[preds[3][i]])

        return f_data, dict(s_pred_data)

    def eval_test(self):
        """Eval all test."""
        self.func_results = []
        self.stmt_results = []
        self.failed = 0
        self.err = []
        for i in tqdm(range(len(self.test_data))):
            try:
                f_ret, s_ret = self.test_item(i)
            except Exception as E:
                self.failed += 1
                self.err.append(E)
                continue
            self.func_results.append(f_ret)
            self.stmt_results.append(s_ret)
        return


if __name__ == "__main__":
    checkpoint = "raytune_-1/202109031655_f87dcf9_add_perfect_test/tune_linevd/train_linevd_2a3f5_00013_13_gatdropout=0.2,gnntype=gat,gtype=pdg+raw,hdropout=0.3,modeltype=gat2layer,stmtweight=10_2021-09-04_07-55-21/checkpoint_epoch=129-step=63310/checkpoint"

    # Load modules
    model = lvd.LitGNN()
    datamodule_args = {"batch_size": 1024, "nsampling_hops": 2, "gtype": "pdg+raw"}
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
    best_model = svd.processed_dir() / checkpoint
    model = lvd.LitGNN.load_from_checkpoint(best_model, strict=False)
    trainer.test(model, data)

    # Eval empirically
    eebv = EmpEvalBigVul(model.all_funcs, data.test)
    eebv.eval_test()

    # Evaluate true-positive method predictions
    # Evaluate false-positive method predictions
    # Evaluate false-negative method predictions
    # Evaluate true-negative method predictions

    # Evaluate true-positive statement predictions
    # Evaluate false-positive statement predictions
    # Evaluate false-negative statement predictions
    # Evaluate true-negative statement predictions
