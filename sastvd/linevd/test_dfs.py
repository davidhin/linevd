from glob import glob

import pandas as pd
import sastvd as svd

# %% RQ2
results = glob(str(svd.outputs_dir() / "rq2/*.csv"))
res_df = pd.concat([pd.read_csv(i) for i in results])
res_df = res_df.sort_values("stmtline_f1", ascending=0)
metrics = [i for i in res_df.columns if "stmtline" in i]
res_df = res_df.groupby("trial_id").head(1)


res_df.groupby(["config/gnntype", "config/gtype"]).mean()[metrics]

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

# %% RQ1
results = glob(str(svd.outputs_dir() / "rq1/*.csv"))
res_df = pd.concat([pd.read_csv(i) for i in results])
res_df = res_df.sort_values("stmt_f1", ascending=0)
metrics = [i for i in res_df.columns if "stmt" in i]
res_df = res_df.groupby("trial_id").head(1)

res_df.groupby("config/embtype").max()[metrics]
res_df.groupby("config/embtype").std()

res_df.columns


print(
    res_df.groupby(["config/feat_type"])
    .head(1)[["config/feat_type"] + metrics]
    .to_latex(index=0)
)
