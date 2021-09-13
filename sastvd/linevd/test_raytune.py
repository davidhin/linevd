from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import sastvd as svd
import sastvd.linevd as lvd
from ray.tune import Analysis

# Default splits
# run_id = "202109061248_c5a12e0_keep_2_checkpoints"  # Additional runs
splits = "default"
run_id = "raytune_-1/202109031655_f87dcf9_add_perfect_test"
# run_id = "202109061634_099b37f_modify_test_models_script"
# run_id = "202109071705_7e3cb4c_update_hparams"
# run_id = "202109080845_d5d33a2_add_optimal_f1"  # Codebert method-only
# run_id = "202109080911_d5d33a2_add_optimal_f1"  # Codebert line-only

# Cross project splits
run_id = "raytune_-1/202109091312_2003c8f_add_cross-project_splits"  # Cross-project detection
run_id = "raytune_crossproject_-1/202109091651_0e864e4_rename_file"  # Cross-project detection

# Load Single Raytune Analysis object
d = svd.processed_dir() / run_id / "tune_linevd"
analysis = Analysis(d)
df = analysis.dataframe().sort_values("val_loss")

# Load Multi Raytune Analysis object
df_list = []
for d in glob(str(svd.processed_dir() / "raytune_features_-1/*")):
    df_list.append(Analysis(d).dataframe())
df = pd.concat(df_list)
run_id = "runid/rq1"

# Get configurations
if "config/splits" not in df.columns:
    df["config/splits"] = "default"
if "config/embtype" not in df.columns:
    df["config/embtype"] = "codebert"
configs = df[["config/gtype", "config/splits", "config/embtype"]]
configs = configs.drop_duplicates().to_dict("records")


def get_relevant_metrics(trial_result):
    """Get relevant metrics from results."""
    ret = {}
    ret["trial_id"] = trial_result[0]
    ret["checkpoint"] = trial_result[1]
    ret["acc@5"] = trial_result[2][5]
    ret["stmt_f1"] = trial_result[3]["f1"]
    ret["stmt_rec"] = trial_result[3]["rec"]
    ret["stmt_prec"] = trial_result[3]["prec"]
    ret["stmt_mcc"] = trial_result[3]["mcc"]
    ret["stmt_fpr"] = trial_result[3]["fpr"]
    ret["stmt_fnr"] = trial_result[3]["fnr"]
    ret["stmt_rocauc"] = trial_result[3]["roc_auc"]
    ret["stmt_prauc"] = trial_result[3]["pr_auc"]
    ret["stmt_prauc_pos"] = trial_result[3]["pr_auc_pos"]
    ret["func_f1"] = trial_result[4]["f1"]
    ret["func_rec"] = trial_result[4]["rec"]
    ret["func_prec"] = trial_result[4]["prec"]
    ret["func_mcc"] = trial_result[4]["mcc"]
    ret["func_fpr"] = trial_result[4]["fpr"]
    ret["func_fnr"] = trial_result[4]["fnr"]
    ret["func_rocauc"] = trial_result[4]["roc_auc"]
    ret["func_prauc"] = trial_result[4]["pr_auc"]
    ret["MAP@5"] = trial_result[5]["MAP@5"]
    ret["nDCG@5"] = trial_result[5]["nDCG@5"]
    ret["MFR"] = trial_result[5]["MFR"]
    ret["MAR"] = trial_result[5]["MAR"]
    ret["stmtline_f1"] = trial_result[6]["f1"]
    ret["stmtline_rec"] = trial_result[6]["rec"]
    ret["stmtline_prec"] = trial_result[6]["prec"]
    ret["stmtline_mcc"] = trial_result[6]["mcc"]
    ret["stmtline_fpr"] = trial_result[6]["fpr"]
    ret["stmtline_fnr"] = trial_result[6]["fnr"]
    ret["stmtline_rocauc"] = trial_result[6]["roc_auc"]
    ret["stmtline_prauc"] = trial_result[6]["pr_auc"]
    ret["stmtline_prauc_pos"] = trial_result[6]["pr_auc_pos"]

    ret = {k: round(v, 3) if isinstance(v, float) else v for k, v in ret.items()}
    return ret


# Get trial results list
trial_results = []
for config in configs:
    df_gtype = df[
        (df["config/gtype"] == config["config/gtype"])
        & (df["config/splits"] == config["config/splits"])
        & (df["config/embtype"] == config["config/embtype"])
    ]
    hparam_cols = df_gtype.columns[df_gtype.columns.str.contains("config")].tolist()
    data = lvd.BigVulDatasetLineVDDataModule(
        batch_size=1024,
        nsampling_hops=2,
        gtype=config["config/gtype"],
        splits=config["config/splits"],
        feat=config["config/embtype"],
    )
    for row in df_gtype.itertuples():
        chkpt_list = glob(row.logdir + "/checkpoint_*")
        chkpt_list = [i + "/checkpoint" for i in chkpt_list]
        for chkpt in chkpt_list:
            model = lvd.LitGNN()
            model = lvd.LitGNN.load_from_checkpoint(chkpt, strict=False)
            trainer = pl.Trainer(gpus=1, default_root_dir="/tmp/")
            trainer.test(model, data)
            res = [
                row.trial_id,
                Path(chkpt).parent.name,
                model.res1vo,
                model.res2mt,
                model.res2f,
                model.res3vo,
                model.res2,
            ]
            trial_results.append(res)

            # Save DF
            res_rows = []
            for tr in trial_results:
                mets = get_relevant_metrics(tr)
                hparams = df[df.trial_id == tr[0]][hparam_cols].to_dict("records")[0]
                res_rows.append({**mets, **hparams})
            res_df = pd.DataFrame.from_records(res_rows)
            res_df.to_csv(svd.outputs_dir() / f"{run_id.split('/')[-1]}.csv", index=0)

# Test components
results = glob(str(svd.outputs_dir() / "*.csv"))
results = [i for i in results if "add_optimal_f1" not in i]
results = [i for i in results if "_val.csv" not in i]
res_df = pd.concat([pd.read_csv(i) for i in results])
res_df["stmtfunc"] = res_df.stmt_f1 + res_df.func_f1
res_df = res_df.sort_values("stmtfunc", ascending=0)
metrics = ["stmt_f1", "stmt_rocauc", "stmt_prauc", "MAP@5", "nDCG@5"]
print(
    res_df.groupby(["config/modeltype"])
    .head(1)[["config/modeltype", "config/gnntype"] + metrics]
    .to_latex(index=0)
)
print(
    res_df.groupby(["config/gtype"])
    .head(1)[["config/gtype"] + metrics]
    .to_latex(index=0)
)
