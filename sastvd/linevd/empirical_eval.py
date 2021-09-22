import os
from collections import Counter, defaultdict
from glob import glob
from math import sqrt

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.helpers.dclass as svddc
import sastvd.linevd as lvd
import sastvd.linevd.c_builtins as cbuiltin
from ray.tune import Analysis
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
            if f_data["pred"] == 1:
                s_pred_data[preds[3][i]]["pred"] = list(preds[0][i])
            else:
                s_pred_data[preds[3][i]]["pred"] = [1, 0]
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

    # Get analysis directories in storage/processed
    raytune_dirs = glob(str(svd.processed_dir() / "raytune_*_-1"))
    tune_dirs = [i for j in [glob(f"{rd}/*") for rd in raytune_dirs] for i in j]

    # Load full dataframe
    df_list = []
    for d in tune_dirs:
        df_list.append(Analysis(d).dataframe())
    df = pd.concat(df_list)
    df = df[df["config/splits"] == "default"]

    # Load results df
    results = glob(str(svd.outputs_dir() / "rq_results_new/*.csv"))
    res_df = pd.concat([pd.read_csv(i) for i in results])

    # Merge DFs and load best model
    mdf = df.merge(res_df[["trial_id", "checkpoint", "stmt_f1"]], on="trial_id")
    bestiloc = 0
    while True:
        best = mdf.sort_values("stmt_f1", ascending=0).iloc[bestiloc]
        best_path = f"{best['logdir']}/{best['checkpoint']}/checkpoint"
        if os.path.exists(best_path):
            break
        else:
            print("Doesn't exist: " + str(best_path))
            bestiloc += 1
            continue

    # Load modules
    model = lvd.LitGNN()
    datamodule_args = {
        "batch_size": 1024,
        "nsampling_hops": 2,
        "gtype": best["config/gtype"],
        "splits": best["config/splits"],
    }
    data = lvd.BigVulDatasetLineVDDataModule(**datamodule_args)
    trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
    model = lvd.LitGNN.load_from_checkpoint(best_path, strict=False)
    trainer.test(model, data)

    # Check statement metrics
    print(model.res2mt)

    # Eval empirically
    eebv = EmpEvalBigVul(model.all_funcs, data.test)
    eebv.eval_test()

    # Func matrix evaluation
    func_tp = [i for i in eebv.func_results if i["pred"] == 1 and i["vul"] == 1]
    func_tn = [i for i in eebv.func_results if i["pred"] == 0 and i["vul"] == 0]
    func_fp = [i for i in eebv.func_results if i["pred"] == 1 and i["vul"] == 0]
    func_fn = [i for i in eebv.func_results if i["pred"] == 0 and i["vul"] == 1]

    # Statement matrix evaluation
    stmt_tp = []
    stmt_tn = []
    stmt_fp = []
    stmt_fn = []
    for func in eebv.stmt_results:
        for stmt in func.values():
            if stmt["pred"][1] > model.f1thresh and stmt["vul"] == 1:
                stmt_tp.append(stmt)
            if stmt["pred"][1] < model.f1thresh and stmt["vul"] == 0:
                stmt_tn.append(stmt)
            if stmt["pred"][1] > model.f1thresh and stmt["vul"] == 0:
                stmt_fp.append(stmt)
            if stmt["pred"][1] < model.f1thresh and stmt["vul"] == 1:
                stmt_fn.append(stmt)

    # Main Analysis - Statement-level
    def count_labels(stmts):
        """Get info about statements."""
        label_info = []
        for info in stmts:
            if info["_label"] == "CALL":
                if "<operator>" in info["name"]:
                    if "assignment" in info["name"]:
                        label_info.append("Assignment Operator")
                        continue
                    if (
                        "addition" in info["name"]
                        or "subtraction" in info["name"]
                        or "division" in info["name"]
                        or "Plus" in info["name"]
                        or "Minus" in info["name"]
                        or "minus" in info["name"]
                        or "plus" in info["name"]
                        or "modulo" in info["name"]
                        or "multiplication" in info["name"]
                    ):
                        label_info.append("Arithmetic Operator")
                        continue
                    if (
                        "lessThan" in info["name"]
                        or "greaterThan" in info["name"]
                        or "EqualsThan" in info["name"]
                        or "equals" in info["name"]
                    ):
                        label_info.append("Comparison Operator")
                        continue
                    if (
                        "FieldAccess" in info["name"]
                        or "IndexAccess" in info["name"]
                        or "fieldAccess" in info["name"]
                        or "indexAccess" in info["name"]
                    ):
                        label_info.append("Access Operator")
                        continue
                    if (
                        "logical" in info["name"]
                        or "<operator>.not" in info["name"]
                        or "<operator>.or" in info["name"]
                        or "<operator>.and" in info["name"]
                        or "conditional" in info["name"]
                    ):
                        label_info.append("Logical Operator")
                        continue
                    if "<operator>.cast" in info["name"]:
                        label_info.append("Cast Operator")
                        continue
                    if "<operator>" in info["name"]:
                        label_info.append("Other Operator")
                        continue
                elif info["name"] in cbuiltin.l_funcs:
                    label_info.append("Builtin Function Call")
                    continue
                else:
                    label_info.append("External Function Call")
                    continue
            if info["_label"] == "CONTROL_STRUCTURE":
                label_info.append(info["controlStructureType"])
                continue
            label_info.append(info["_label"])
        return Counter(label_info)

    def calc_mcc(tp, fp, tn, fn):
        """Generalised MCC."""
        x = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return ((tp * tn) - (fp * fn)) / sqrt(x)

    def calc_f1(tp, fp, tn, fn):
        """Generalised F1."""
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * ((prec * rec) / (prec + rec))

    counts = [
        count_labels(stmt_fp),
        count_labels(stmt_fn),
        count_labels(stmt_tp),
        count_labels(stmt_tn),
    ]
    all_node_types = set([j for k in [list(i.keys()) for i in counts] for j in k])
    ntmat = []
    for node_type in all_node_types:
        fp = counts[0][node_type] if node_type in counts[0] else 0
        fn = counts[1][node_type] if node_type in counts[1] else 0
        tp = counts[2][node_type] if node_type in counts[2] else 0
        tn = counts[3][node_type] if node_type in counts[3] else 0
        try:
            mcc = calc_mcc(tp, fp, tn, fn)
        except:
            mcc = None
        try:
            f1 = calc_f1(tp, fp, tn, fn)
        except:
            f1 = None
        if node_type == "METHOD_PARAMETER_IN":
            node_type = "Parameter In"
        if node_type == "METHOD_PARAMETER_OUT":
            node_type = "Parameter Out"
        if node_type == "ThrowStatement":
            node_type = "Throw Statement"
        if node_type == "JUMP_TARGET":
            node_type = "Jump Target"
        if node_type == "FIELD_IDENTIFIER":
            node_type = "Field Identifier"
        if node_type == "ThrowStatement":
            continue
        ntmat.append(
            {
                "Statement Type": node_type.title(),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                # "MCC": abs(round(mcc, 2)) if mcc else None,
                "F1": abs(round(f1, 2)) if f1 else None,
            }
        )

    # Get final dataframe
    stmt_analysis = pd.DataFrame.from_records(ntmat).sort_values("F1", ascending=0)
    print(stmt_analysis.to_latex(index=0))

    # Func analysis: length of function, number of interprocedural calls, other software metrics.
