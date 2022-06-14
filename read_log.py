import sys
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

currows = []

for l in sys.stdin:
    if l.startswith("newffffffile"):
        curfeat = None
        curseed = None
        curckpt = None
        # db = {}
        # currow = []
        # ckpts = []
    # elif l.startswith("endffffffile"):
    #     # for runname, data in db.items():
    #     #     print(",".join(runname))
    #     #     print("checkpoint", *ckpts, sep=",")
    #     #     for metricname, datas in data.items():
    #     #         print(metricname, *datas, sep=",")
    #     df = pd.concat((df, pd.DataFrame(currows)), ignore_index=True)
    elif l.startswith("flow_gnn_MSR"):
        m = re.match(r"flow_gnn_MSR_graph_(.*)_None_None.*256_([0-4]).*/checkpoints/(.*).ckpt", l)
        curfeat = m.group(1)
        curseed = m.group(2)
        curckpt = m.group(3)
        curckpt = re.sub(r"periodical-([0-9]{2}-)", r"periodical-0\1", curckpt)
    elif l.startswith("test_"):
        metricname, metric = l.split()
        # if m.group(3) not in ckpts:
        #     ckpts.append(m.group(3))
        # # print(curname, curseed, metricname, metric)
        # if _id not in db:
        #     db[_id] = {}
        # dbtbl = db[_id]
        # if metricname not in dbtbl:
        #     dbtbl[metricname] = []
        # dbtbl[metricname].append(metric)
        currows.append({
            "feat": curfeat,
            "seed": curseed,
            "ckpt": curckpt,
            "metric": metricname,
            "value": float(metric),
        })
df = pd.DataFrame(currows)
# print(df)
print(df[df["metric"] == "test_class_f1_epoch"])
df.to_csv("data.csv")

df["val_loss"] = df["ckpt"].str.extract(r".*val_loss=([0-9]+.[0-9]+)").astype(float)
df = df.dropna(subset=["val_loss"])
df.to_csv("data_with_val_loss.csv")

df = df[["feat", "ckpt", "seed", "metric", "value"]]
df = df[df["ckpt"].str.startswith("performance-")]
df["val_loss"] = df["ckpt"].str.extract(r".*val_loss=([0-9]+.[0-9]+)").astype(float)
df = df[df["metric"] == "test_class_f1_epoch"]
# for feat, group in df.groupby("feat"):
# print(feat)
print(df)
sns.lineplot(data=df, x="val_loss", y="value", hue="feat")
# plt.title(feat)
plt.savefig(f"val_loss.png")
plt.tight_layout()
plt.close()
