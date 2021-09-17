from glob import glob

import pandas as pd
import sastvd as svd

pd.set_option("display.max_columns", None)

# %% Phoenix
results = glob(str(svd.outputs_dir() / "phoenix/rq_results/*.csv"))
results2 = glob(str(svd.outputs_dir() / "phoenix_new/rq_results_new/*.csv"))
results += results2
res_df = pd.concat([pd.read_csv(i) for i in results])
res_df = res_df.drop_duplicates(["trial_id", "checkpoint"])
metrics = [i for i in res_df.columns if "stmt" in i]
metricsf = [i for i in res_df.columns if "func" in i]
rankedcols = ["acc@5", "MAP@5", "nDCG@5", "MFR"]
metricsline = [i for i in res_df.columns if "stmtline" in i]
configcols = [i for i in res_df.columns if "config" in i]
res_df["config/gtype"] = res_df["config/gtype"].apply(lambda x: x.replace("+raw", ""))

# RQ2 setup
res_df["config/gnntype"] = res_df.apply(
    lambda x: "nognn" if x["config/modeltype"] == "mlponly" else x["config/gnntype"],
    axis=1,
)
res_df["config/gtype"] = res_df.apply(
    lambda x: "nognn" if x["config/modeltype"] == "mlponly" else x["config/gtype"],
    axis=1,
)

# RQ1
rq1_cg = "config/embtype"
rq1 = res_df.sort_values("stmtline_f1", ascending=0).groupby("trial_id").head(1)
rq1 = rq1[rq1["config/splits"] == "default"]
rq1 = rq1.groupby(rq1_cg).head(5).groupby(rq1_cg).mean()[metricsline]

# RQ2
rq2_cg = ["config/gnntype", "config/gtype"]
rq2 = res_df[res_df["config/splits"] == "default"]
rq2a = rq2.sort_values("stmtline_f1", ascending=0).groupby("trial_id").head(1)
rq2b = rq2.sort_values("stmt_f1", ascending=0).groupby("trial_id").head(1)
rq2b = rq2b[rq2b["config/multitask"] == "linemethod"]
rq2a = rq2a.groupby(rq2_cg).head(5).groupby(rq2_cg).mean()[metricsline]
rq2a.columns = [i.replace("line", "") for i in rq2a.columns]
rq2b = rq2b.groupby(rq2_cg).head(5).groupby(rq2_cg).mean()[metrics]
rq2a["multitask"] = "line"
rq2b["multitask"] = "line+method"
rq2final = pd.concat([rq2a, rq2b]).reset_index().groupby(rq2_cg + ["multitask"]).sum()


# RQ3
rq3_cg = "config/multitask"
rq3a = res_df.sort_values("stmtline_f1", ascending=0).groupby("trial_id").head(1)
rq3b = res_df.sort_values("stmt_f1", ascending=0).groupby("trial_id").head(1)
rq3a = rq3a[rq3a["config/splits"] == "default"]
rq3b = rq3b[rq3b["config/splits"] == "default"]
rq3a = rq3a.head(5).groupby(lambda x: True).mean()[metricsline]
rq3a.columns = [i.replace("line", "") for i in rq3a.columns]
rq3b = rq3b.head(5).groupby(lambda x: True).mean()
rq3 = pd.concat([rq3a, rq3b[metrics]])

# RQ5
rq5_cg = "config/splits"
rq5 = res_df.sort_values("stmt_f1", ascending=0).groupby("trial_id").head(1)
rq5 = rq5.groupby(rq5_cg).head(5).groupby(rq5_cg).mean()[metrics + ["MFR"]]

# Latex
rq1.round(3)[metricsline]
rq2final.round(3)[metrics]
rq3.round(3)[metrics]
rq3b.round(3)[rankedcols]
rq3b.round(3)[metricsf]
print(
    rq5.round(3)[metrics][
        ["stmt_f1", "stmt_rec", "stmt_prec", "stmt_rocauc", "stmt_prauc"]
    ].to_latex()
)
